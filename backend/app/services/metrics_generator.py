import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def generate_metrics(original, enhanced):
    if original is None or enhanced is None:
        raise ValueError("Both original and enhanced images are required.")

    if original.shape != enhanced.shape:
        raise ValueError("Images must have matching dimensions and channels.")

    original = np.asarray(original, dtype=np.uint8)
    enhanced = np.asarray(enhanced, dtype=np.uint8)

    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    psnr = float(
        peak_signal_noise_ratio(original, enhanced, data_range=255)
    )

    ssim = float(
        structural_similarity(
            original_gray,
            enhanced_gray,
            data_range=255,
        )
    )

    original_sharpness = float(
        cv2.Laplacian(original_gray, cv2.CV_64F).var()
    )
    enhanced_sharpness = float(
        cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
    )

    return {
        "psnr_db": round(psnr, 3),
        "ssim": round(ssim, 4),
        "original_sharpness": round(original_sharpness, 2),
        "enhanced_sharpness": round(enhanced_sharpness, 2),
        "brightness_before": round(float(original_gray.mean()), 2),
        "brightness_after": round(float(enhanced_gray.mean()), 2),
    }