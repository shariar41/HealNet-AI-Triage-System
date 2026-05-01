import pandas as pd
import matplotlib.pyplot as plt

from config import MENDELEY_RAW_CSV, MENDELEY_FEATURES_CSV, FIGURE_DIR
from training.triage_mapping import DISEASE_TO_TRIAGE

def detect_disease_column(df: pd.DataFrame) -> str:
    possible_names = ["diseases", "disease", "Disease", "prognosis", "label"]
    for col in possible_names:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find disease column. Columns found: {list(df.columns)}")

def main():
    df = pd.read_csv(MENDELEY_RAW_CSV)

    disease_col = detect_disease_column(df)
    print("Detected disease column:", disease_col)

    df[disease_col] = df[disease_col].astype(str).str.strip().str.lower()
    df = df[df[disease_col].isin(DISEASE_TO_TRIAGE.keys())].copy()

    print("After filtering:", len(df))

    df["triage_label"] = df[disease_col].map(DISEASE_TO_TRIAGE).astype(int)
    df = df.rename(columns={disease_col: "disease"})

    df.to_csv(MENDELEY_FEATURES_CSV, index=False)
    print("Saved:", MENDELEY_FEATURES_CSV)

    counts = df["triage_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Home", "Doctor", "Emergency"], [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)])
    plt.title("Mendeley Training Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/mendeley_class_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()