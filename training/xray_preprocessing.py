import cv2
import numpy as np
from PIL import Image

def preprocess_xray_for_model(image_pil, image_size=(224, 224)):
    image_np = np.array(image_pil)

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    gray = cv2.resize(gray, image_size)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    stacked = np.stack([enhanced, blurred, edges], axis=-1)

    return stacked.astype(np.uint8)

def preprocess_xray_from_path(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    return preprocess_xray_for_model(image, image_size=image_size)