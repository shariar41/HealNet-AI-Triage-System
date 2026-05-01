import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from training.xray_preprocessing import preprocess_xray_for_model

def main():
    image_path = r"data/raw/chest_xray/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"

    image = Image.open(image_path).convert("RGB")
    processed = preprocess_xray_for_model(image)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("Processed 3-channel image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()