import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import os

from app.ml.enhancement import enhance_image

# 🔥 Load pretrained model (updated API, no warning)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 🔥 Correct transform (IMPORTANT: normalization added)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image_np):
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    confidence = torch.max(probs).item()
    return confidence


def evaluate(image_path):
    # 🔥 Build absolute path (stable, no path bugs)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    full_path = os.path.join(base_dir, image_path)

    img = cv2.imread(full_path)

    if img is None:
        raise ValueError(f"Image not found: {full_path}")

    # 🔹 Original
    original_conf = predict(img)

    # 🔹 Enhanced
    with open(full_path, "rb") as f:
        image_bytes = f.read()

    _, enhanced, _ = enhance_image(image_bytes)
    enhanced_conf = predict(enhanced)

    # 🔹 Compare
    improvement = enhanced_conf - original_conf

    print("\n---", image_path, "---")
    print("Original Confidence :", round(original_conf, 4))
    print("Enhanced Confidence:", round(enhanced_conf, 4))
    print("Delta              :", round(improvement, 4))

    if improvement > 0.02:
        print("✔ Improvement")
    elif improvement < -0.02:
        print("⚠ Degradation")
    else:
        print("~ No significant change")


# 🔥 Batch evaluation (IMPORTANT — no single-image bias)
if __name__ == "__main__":
    images = [
        "sample.jpg",
        "data/img1.jpg",
        "data/img2.jpg"
    ]

    for img_path in images:
        evaluate(img_path)