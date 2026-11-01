import cv2
import numpy as np


def analyze_quality(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness using Laplacian variance
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = gray.std()

    # Quality score
    quality_score = (
        (sharpness * 0.5) +
        (contrast * 0.3) +
        (brightness * 0.2)
    ) / 10

    return {
        "sharpness": float(sharpness),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "quality_score": float(quality_score)
    }