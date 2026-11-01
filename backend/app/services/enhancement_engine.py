import cv2
import numpy as np


def enhance_image(image, quality=None):
    """
    Input and output are RGB NumPy arrays.
    Uses conservative enhancement based on basic quality measurements.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if quality is None:
        from app.services.quality_analyzer import analyze_quality
        quality = analyze_quality(image)

    output = image.copy()

    brightness = quality["brightness"]
    contrast = quality["contrast"]
    noise = quality["noise_estimate"]

    # Correct exposure only when clearly needed.
    if brightness < 60 or brightness > 200:
        lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
        lightness, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )
        lightness = clahe.apply(lightness)

        output = cv2.cvtColor(
            cv2.merge((lightness, a, b)),
            cv2.COLOR_LAB2RGB,
        )

    # Apply mild denoising only when estimated noise is high.
    if noise > 20:
        output = cv2.fastNlMeansDenoisingColored(
            output, None, 3, 3, 7, 21
        )

    # Mild contrast correction for low-contrast images.
    if contrast < 25:
        output = cv2.convertScaleAbs(
            output, alpha=1.05, beta=0
        )

    return output