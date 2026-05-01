import os
import sys
import re
import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MENDELEY_RAW_CSV,
    SYMPTOM_MODEL_V2_PATH,
    SYMPTOM_FEATURE_COLUMNS_PATH,
    IMAGE_MODEL_PATH,
    DQN_MODEL_PATH
)
from models.image_model import ChestXrayResNet
from models.dqn_agent import DQN
from training.xray_preprocessing import preprocess_xray_for_model
from training.safety_rules import emergency_keyword_score
from utils import softmax_numpy, triage_label_to_name, xray_label_to_name, confidence_from_probs

st.set_page_config(page_title="HealNet RL-Triage", layout="centered")

def detect_disease_column(df):
    for col in ["diseases", "disease", "Disease", "prognosis", "label"]:
        if col in df.columns:
            return col
    raise ValueError("Disease column not found")

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

@st.cache_resource
def load_symptom_artifacts():
    model = joblib.load(SYMPTOM_MODEL_V2_PATH)
    feature_cols = joblib.load(SYMPTOM_FEATURE_COLUMNS_PATH)

    mend_df = joblib.load if False else None  # no-op to keep structure simple
    return model, feature_cols

@st.cache_resource
def load_mendeley_symptom_columns():
    import pandas as pd
    mend_df = pd.read_csv(MENDELEY_RAW_CSV)
    disease_col = detect_disease_column(mend_df)
    symptom_cols = [c for c in mend_df.columns if c != disease_col]
    return symptom_cols

@st.cache_resource
def load_image_model():
    model = ChestXrayResNet(num_classes=2)
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_dqn_model():
    model = DQN(state_dim=3, action_dim=3)
    model.load_state_dict(torch.load(DQN_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def build_symptom_feature_vector(text, feature_cols, symptom_cols):
    text = clean_text(text)
    feature_dict = {}

    for col in feature_cols:
        symptom_clean = clean_text(col)
        feature_dict[col] = 1 if symptom_clean in text else 0

    return np.array([feature_dict[col] for col in feature_cols], dtype=np.float32).reshape(1, -1)

def predict_symptoms(symptom_text, model, feature_cols, symptom_cols):
    x = build_symptom_feature_vector(symptom_text, feature_cols, symptom_cols)
    probs = model.predict_proba(x)[0]
    pred = int(np.argmax(probs))
    return probs, pred

def predict_xray(image_pil, model):
    processed_img = preprocess_xray_for_model(image_pil, image_size=(224, 224))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(processed_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor).numpy()[0]

    probs = softmax_numpy(logits)
    pred = int(np.argmax(probs))
    return processed_img, probs, pred

def apply_safety_fusion(symptom_probs, symptom_text, xray_probs=None):
    fused = symptom_probs.copy()

    score = emergency_keyword_score(symptom_text)
    if score >= 2:
        fused[2] += 0.20
        fused[0] -= 0.10
    elif score == 1:
        fused[2] += 0.10

    if xray_probs is not None:
        fused[2] += 0.60 * xray_probs[1]
        fused[0] -= 0.30 * xray_probs[1]

    fused = np.clip(fused, 0.001, None)
    fused = fused / fused.sum()
    return fused

def predict_final_action(fused_probs, dqn_model):
    state = torch.tensor(fused_probs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn_model(state)
    action = int(torch.argmax(q_values, dim=1).item())
    return action, q_values.numpy()[0]

st.title("HealNet RL-Triage")
st.write("Smart healthcare triage MVP using symptom intelligence, advanced X-ray preprocessing, and reinforcement learning.")

symptom_text = st.text_area(
    "Enter symptoms",
    placeholder="Example: chest pain, shortness of breath, dizziness, fever"
)

uploaded_image = st.file_uploader(
    "Optional Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if st.button("Analyze"):
    if not symptom_text.strip():
        st.error("Please enter symptoms.")
        st.stop()

    symptom_model, feature_cols = load_symptom_artifacts()
    symptom_cols = load_mendeley_symptom_columns()
    dqn_model = load_dqn_model()

    symptom_probs, symptom_pred = predict_symptoms(symptom_text, symptom_model, feature_cols, symptom_cols)

    st.subheader("Symptom Model Output")
    st.write({
        "Home Care Probability": float(symptom_probs[0]),
        "Doctor Visit Soon Probability": float(symptom_probs[1]),
        "Emergency Probability": float(symptom_probs[2])
    })

    xray_probs = None

    if uploaded_image is not None:
        try:
            image_model = load_image_model()
            image = Image.open(uploaded_image).convert("RGB")
            processed_img, xray_probs, xray_pred = predict_xray(image, image_model)

            st.subheader("Chest X-ray Model Output")
            st.image(image, caption="Original Uploaded X-ray", use_container_width=True)
            st.image(processed_img, caption="Processed X-ray Representation", use_container_width=True)

            st.write({
                "Predicted X-ray Class": xray_label_to_name(xray_pred),
                "Normal Probability": float(xray_probs[0]),
                "Pneumonia Probability": float(xray_probs[1])
            })

        except Exception as e:
            st.warning(f"Image model could not process the image: {e}")

    fused_probs = apply_safety_fusion(symptom_probs, symptom_text, xray_probs)
    final_action, q_values = predict_final_action(fused_probs, dqn_model)

    st.subheader("Final Fused Decision")
    st.write({
        "Fused Home Probability": float(fused_probs[0]),
        "Fused Doctor Probability": float(fused_probs[1]),
        "Fused Emergency Probability": float(fused_probs[2])
    })

    st.subheader("Final RL Triage Decision")
    st.success(triage_label_to_name(final_action))

    st.write("Confidence:", round(confidence_from_probs(fused_probs) * 100, 2), "%")
    st.write("Q-values:", q_values.tolist())
    st.write("Emergency keyword score:", emergency_keyword_score(symptom_text))

    if final_action == 2:
        st.error("Recommendation: Seek urgent medical attention.")
    elif final_action == 1:
        st.warning("Recommendation: Schedule a doctor visit soon.")
    else:
        st.info("Recommendation: Home care may be sufficient, but monitor symptoms.")