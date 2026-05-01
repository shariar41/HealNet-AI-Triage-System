import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# Change this if your actual Mendeley file has a different name
MENDELEY_RAW_CSV = os.path.join(RAW_DIR, "mendeley", "disease_symptoms.csv")
MENDELEY_FEATURES_CSV = os.path.join(PROCESSED_DIR, "mendeley_features.csv")

SYMPTOM_RAW_CSV = os.path.join(RAW_DIR, "symptom2disease", "Symptom2Disease.csv")
SYMPTOM_FEATURES_CSV = os.path.join(PROCESSED_DIR, "symptom2disease_features.csv")

CHEST_XRAY_DIR = os.path.join(RAW_DIR, "chest_xray")
CHEST_XRAY_TRAIN = os.path.join(CHEST_XRAY_DIR, "train")
CHEST_XRAY_VAL = os.path.join(CHEST_XRAY_DIR, "val")
CHEST_XRAY_TEST = os.path.join(CHEST_XRAY_DIR, "test")

SYMPTOM_MODEL_V2_PATH = os.path.join(CHECKPOINT_DIR, "symptom_model_v2.joblib")
SYMPTOM_FEATURE_COLUMNS_PATH = os.path.join(CHECKPOINT_DIR, "symptom_feature_columns.joblib")

IMAGE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "image_model.pt")

RL_DATASET_PATH = os.path.join(PROCESSED_DIR, "rl_dataset.csv")
DQN_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_agent.pt")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)