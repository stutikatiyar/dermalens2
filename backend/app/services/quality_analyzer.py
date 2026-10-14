import cv2
import numpy as np


def analyze_quality(image):
    """
    Input: RGB NumPy image.
    Output: interpretable image-quality measurements.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Higher variance often indicates a sharper image.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # A rough noise estimate; not a ground-truth noise measurement.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    noise_estimate = float(np.std(residual))

    issues = []

    if sharpness < 50:
        issues.append("possible_blur")

    if brightness < 60:
        issues.append("underexposure")
    elif brightness > 200:
        issues.append("overexposure")

    if contrast < 25:
        issues.append("low_contrast")

    if min(width, height) < 256:
        issues.append("low_resolution")

    if noise_estimate > 20:
        issues.append("possible_noise")

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise_estimate": round(noise_estimate, 2),
        "issues": issues,
    }