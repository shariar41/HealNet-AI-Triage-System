import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from config import CHEST_XRAY_VAL, CHEST_XRAY_TEST, IMAGE_MODEL_PATH, FIGURE_DIR, REPORT_DIR
from models.image_model import ChestXrayResNet
from training.xray_preprocessing import preprocess_xray_from_path, preprocess_xray_for_model
from utils import softmax_numpy

class CustomXrayDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            print(f"Folder not found: {root_dir}")
            return

        for folder in os.listdir(root_dir):
            folder_upper = folder.upper()
            if folder_upper == "NORMAL":
                self.class_to_idx[folder] = 0
            elif folder_upper == "PNEUMONIA":
                self.class_to_idx[folder] = 1

        for class_folder, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_folder)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(class_dir, filename), label))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        processed_img = preprocess_xray_from_path(image_path, image_size=(224, 224))
        tensor_img = self.transform(processed_img)
        return tensor_img, label, image_path, processed_img

def load_model():
    model = ChestXrayResNet(num_classes=2)
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def save_original_vs_processed_examples(dataset, num_examples=3):
    chosen = random.sample(dataset.samples, min(num_examples, len(dataset.samples)))

    for i, (image_path, label) in enumerate(chosen):
        original = Image.open(image_path).convert("RGB")
        processed = preprocess_xray_from_path(image_path, image_size=(224, 224))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap="gray")
        plt.title("Original X-ray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(processed)
        plt.title("Processed X-ray")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/xray_original_vs_processed_{i+1}.png")
        plt.close()

def evaluate_and_save_results(dataset, dataset_name="val"):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    model = load_model()

    all_true = []
    all_pred = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths, _ in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = np.argmax(probs, axis=1)

            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_probs.extend(probs)
            all_paths.extend(paths)

    report = classification_report(all_true, all_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0)

    with open(f"{REPORT_DIR}/xray_{dataset_name}_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(all_true, all_pred)))

    disp = ConfusionMatrixDisplay(confusion_matrix(all_true, all_pred), display_labels=["NORMAL", "PNEUMONIA"])
    disp.plot()
    plt.title(f"X-ray {dataset_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_{dataset_name}_confusion_matrix.png")
    plt.close()

    return all_true, all_pred, all_probs, all_paths

def save_sample_predictions(dataset, all_true, all_pred, all_probs, all_paths, dataset_name="val", num_samples=6):
    indices = list(range(len(all_paths)))
    chosen = random.sample(indices, min(num_samples, len(indices)))

    plt.figure(figsize=(12, 8))

    for plot_idx, idx in enumerate(chosen, start=1):
        image_path = all_paths[idx]
        true_label = all_true[idx]
        pred_label = all_pred[idx]
        probs = all_probs[idx]

        original = Image.open(image_path).convert("RGB")

        plt.subplot(2, 3, plot_idx)
        plt.imshow(original, cmap="gray")
        plt.axis("off")
        plt.title(
            f"True: {'NORMAL' if true_label == 0 else 'PNEUMONIA'}\n"
            f"Pred: {'NORMAL' if pred_label == 0 else 'PNEUMONIA'}\n"
            f"Pneu Prob: {probs[1]:.2f}"
        )

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_{dataset_name}_sample_predictions.png")
    plt.close()

def save_misclassified_examples(all_true, all_pred, all_probs, all_paths, dataset_name="val", num_samples=6):
    misclassified = [i for i in range(len(all_true)) if all_true[i] != all_pred[i]]

    if len(misclassified) == 0:
        print(f"No misclassified samples found for {dataset_name}.")
        return

    chosen = random.sample(misclassified, min(num_samples, len(misclassified)))

    plt.figure(figsize=(12, 8))

    for plot_idx, idx in enumerate(chosen, start=1):
        image_path = all_paths[idx]
        true_label = all_true[idx]
        pred_label = all_pred[idx]
        probs = all_probs[idx]

        original = Image.open(image_path).convert("RGB")

        plt.subplot(2, 3, plot_idx)
        plt.imshow(original, cmap="gray")
        plt.axis("off")
        plt.title(
            f"True: {'NORMAL' if true_label == 0 else 'PNEUMONIA'}\n"
            f"Pred: {'NORMAL' if pred_label == 0 else 'PNEUMONIA'}\n"
            f"Pneu Prob: {probs[1]:.2f}"
        )

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_{dataset_name}_misclassified_examples.png")
    plt.close()

def main():
    print("Loading validation dataset...")
    val_dataset = CustomXrayDataset(CHEST_XRAY_VAL)
    print("Validation images:", len(val_dataset))

    print("Loading test dataset...")
    test_dataset = CustomXrayDataset(CHEST_XRAY_TEST)
    print("Test images:", len(test_dataset))

    if len(val_dataset) > 0:
        save_original_vs_processed_examples(val_dataset, num_examples=3)

        val_true, val_pred, val_probs, val_paths = evaluate_and_save_results(val_dataset, dataset_name="val")
        save_sample_predictions(val_dataset, val_true, val_pred, val_probs, val_paths, dataset_name="val", num_samples=6)
        save_misclassified_examples(val_true, val_pred, val_probs, val_paths, dataset_name="val", num_samples=6)

    if len(test_dataset) > 0:
        test_true, test_pred, test_probs, test_paths = evaluate_and_save_results(test_dataset, dataset_name="test")
        save_sample_predictions(test_dataset, test_true, test_pred, test_probs, test_paths, dataset_name="test", num_samples=6)
        save_misclassified_examples(test_true, test_pred, test_probs, test_paths, dataset_name="test", num_samples=6)

    print("All analysis images and reports have been saved.")

if __name__ == "__main__":
    main()