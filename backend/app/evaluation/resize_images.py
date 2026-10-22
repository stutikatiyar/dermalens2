import os
import cv2

folder = "app/evaluation/benchmark_images"

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    img = cv2.imread(path)
    if img is not None and (img.shape[0] > 1024 or img.shape[1] > 1024):
        resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, resized)
        print(f"Resized: {f}")

print("Done resizing high-resolution smartphone images.")