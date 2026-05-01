import joblib
import numpy as np
import pandas as pd
from config import MENDELEY_FEATURES_CSV, RL_DATASET_PATH, SYMPTOM_MODEL_V2_PATH, SYMPTOM_FEATURE_COLUMNS_PATH

def main():
    df = pd.read_csv(MENDELEY_FEATURES_CSV)

    model = joblib.load(SYMPTOM_MODEL_V2_PATH)
    feature_cols = joblib.load(SYMPTOM_FEATURE_COLUMNS_PATH)

    X = df[feature_cols].values
    probs = model.predict_proba(X)

    rl_df = pd.DataFrame({
        "p_home": probs[:, 0],
        "p_doctor": probs[:, 1],
        "p_emergency": probs[:, 2],
        "correct_action": df["triage_label"].values
    })

    rl_df.to_csv(RL_DATASET_PATH, index=False)
    print("Saved RL dataset:", RL_DATASET_PATH)
    print(rl_df.head())

if __name__ == "__main__":
    main()