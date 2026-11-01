import cv2
import os
import numpy as np

# base directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
input_path = os.path.join(base_dir, "sample.jpg")

img = cv2.imread(input_path)

if img is None:
    raise ValueError("sample.jpg not found")

# 🔥 create bad versions

# 1. Blur
blur = cv2.GaussianBlur(img, (9, 9), 3)
cv2.imwrite(os.path.join(base_dir, "data/bad_blur.jpg"), blur)

# 2. Low light
dark = (img * 0.4).astype("uint8")
cv2.imwrite(os.path.join(base_dir, "data/bad_dark.jpg"), dark)

# 3. Noise
noise = img.copy()
noise = noise + np.random.normal(0, 25, noise.shape).astype("uint8")
cv2.imwrite(os.path.join(base_dir, "data/bad_noise.jpg"), noise)

print("Bad images created ✔")