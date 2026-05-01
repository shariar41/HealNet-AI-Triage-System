import numpy as np

TRIAGE_LABELS = {
    0: "Home Care",
    1: "Doctor Visit Soon",
    2: "Emergency"
}

XRAY_LABELS = {
    0: "Normal",
    1: "Pneumonia"
}

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / exp_vals.sum()

def triage_label_to_name(label: int) -> str:
    return TRIAGE_LABELS.get(label, "Unknown")

def xray_label_to_name(label: int) -> str:
    return XRAY_LABELS.get(label, "Unknown")

def confidence_from_probs(probs):
    return float(np.max(probs))