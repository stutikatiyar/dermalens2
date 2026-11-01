from app.ml.enhancement import enhance_image
from app.ml.metrics import (
    calculate_sharpness,
    calculate_contrast,
    calculate_brightness,
    brightness_improvement,
    quality_score,
)
from app.services.image_service import ImageService
from app.core.logger import get_logger

logger = get_logger(__name__)

def _number(function, *args):
    try:
        value = function(*args)
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        logger.exception("Metric calculation failed: %s", function.__name__)
        return None

def process_image(image):
    if image is None:
        raise ValueError("No image was supplied.")

    original = image.copy()
    enhanced = enhance_image(original.copy())

    if enhanced is None:
        raise RuntimeError("The enhancement engine returned no image.")

    metrics = {
        "original_sharpness": _number(calculate_sharpness, original),
        "enhanced_sharpness": _number(calculate_sharpness, enhanced),
        "original_contrast": _number(calculate_contrast, original),
        "enhanced_contrast": _number(calculate_contrast, enhanced),
        "original_brightness": _number(calculate_brightness, original),
        "enhanced_brightness": _number(calculate_brightness, enhanced),
        "brightness_improvement": _number(
            brightness_improvement, original, enhanced
        ),
        "quality_score": _number(quality_score, enhanced),
    }

    return {
        "success": True,
        "image": ImageService.encode_base64(enhanced),
        "mode": "enhancement",
        "metrics": metrics,
        "details": {
            "original_dimensions": {
                "width": int(original.shape[1]),
                "height": int(original.shape[0]),
            },
            "enhanced_dimensions": {
                "width": int(enhanced.shape[1]),
                "height": int(enhanced.shape[0]),
            },
        },
    }
