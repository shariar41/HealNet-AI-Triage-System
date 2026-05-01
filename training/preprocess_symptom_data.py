import pandas as pd
from config import SYMPTOM_RAW_CSV, SYMPTOM_TEST_PROCESSED_CSV
from training.triage_mapping import DISEASE_TO_TRIAGE

def main():
    df = pd.read_csv(SYMPTOM_RAW_CSV)

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df["triage_label"] = df["label"].map(DISEASE_TO_TRIAGE)
    df = df.dropna(subset=["triage_label"]).copy()
    df["triage_label"] = df["triage_label"].astype(int)

    df = df.rename(columns={"text": "symptom_text", "label": "disease"})
    df = df[["symptom_text", "disease", "triage_label"]]

    print(df.head())
    print(df["triage_label"].value_counts())

    df.to_csv(SYMPTOM_TEST_PROCESSED_CSV, index=False)
    print("Saved:", SYMPTOM_TEST_PROCESSED_CSV)

    # # sample 2 rows per triage_label
    # sampled_df = (
    # df.groupby("triage_label", group_keys=False)
    #   .apply(lambda x: x.sample(n=min(len(x), 2), random_state=42))
    # )
    # # print results
    # print(sampled_df)
    # print(sampled_df["triage_label"].value_counts())


if __name__ == "__main__":
    main()