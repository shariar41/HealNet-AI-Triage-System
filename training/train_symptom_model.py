import re
import pandas as pd
import matplotlib.pyplot as plt

from config import MENDELEY_RAW_CSV, SYMPTOM_RAW_CSV, SYMPTOM_FEATURES_CSV, FIGURE_DIR
from training.triage_mapping import DISEASE_TO_TRIAGE

def detect_disease_column(df: pd.DataFrame) -> str:
    possible_names = ["diseases", "disease", "Disease", "prognosis", "label"]
    for col in possible_names:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find disease column. Columns found: {list(df.columns)}")

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def main():
    mend_df = pd.read_csv(MENDELEY_RAW_CSV)
    disease_col = detect_disease_column(mend_df)
    symptom_cols = [c for c in mend_df.columns if c != disease_col]

    df = pd.read_csv(SYMPTOM_RAW_CSV)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].apply(clean_text)

    df["triage_label"] = df["label"].map(DISEASE_TO_TRIAGE)
    df = df.dropna(subset=["triage_label"]).copy()
    df["triage_label"] = df["triage_label"].astype(int)

    feature_rows = []

    for _, row in df.iterrows():
        text = row["text"]
        feature_dict = {
            "disease": row["label"],
            "triage_label": row["triage_label"],
            "raw_text": text
        }

        for symptom in symptom_cols:
            symptom_clean = clean_text(symptom)
            feature_dict[symptom] = 1 if symptom_clean in text else 0

        feature_rows.append(feature_dict)

    out_df = pd.DataFrame(feature_rows)
    out_df.to_csv(SYMPTOM_FEATURES_CSV, index=False)
    print("Saved:", SYMPTOM_FEATURES_CSV)

    counts = out_df["triage_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Home", "Doctor", "Emergency"], [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)])
    plt.title("Symptom2Disease External Test Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/symptom2disease_class_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()