import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CHEST_XRAY_TRAIN, CHEST_XRAY_VAL, IMAGE_MODEL_PATH, FIGURE_DIR, REPORT_DIR
from models.image_model import ChestXrayResNet
from training.xray_preprocessing import preprocess_xray_from_path

class CustomXrayDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])
        else:
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
        return tensor_img, label

def save_sample_preprocessing(train_dir):
    sample_path = None
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
            if files:
                sample_path = os.path.join(cls_dir, files[0])
                break

    if sample_path is None:
        return

    original = Image.open(sample_path).convert("RGB")
    processed = preprocess_xray_from_path(sample_path)

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
    plt.savefig(f"{FIGURE_DIR}/xray_preprocessing_example.png")
    plt.close()

def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds, classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CustomXrayDataset(CHEST_XRAY_TRAIN, augment=True)
    val_dataset = CustomXrayDataset(CHEST_XRAY_VAL, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = ChestXrayResNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 8
    losses = []
    train_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total if total > 0 else 0.0
        losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    y_true, y_pred, report = evaluate_model(model, val_loader, device, ["NORMAL", "PNEUMONIA"])
    print("\nValidation Report:")
    print(report)

    with open(f"{REPORT_DIR}/xray_model_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.title("X-ray Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_training_loss.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_accs, marker="o")
    plt.title("X-ray Model Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_training_accuracy.png")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["NORMAL", "PNEUMONIA"])
    disp.plot()
    plt.title("X-ray Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/xray_validation_confusion_matrix.png")
    plt.close()

    save_sample_preprocessing(CHEST_XRAY_TRAIN)

    torch.save(model.state_dict(), IMAGE_MODEL_PATH)
    print("Saved image model to:", IMAGE_MODEL_PATH)

if __name__ == "__main__":
    main()