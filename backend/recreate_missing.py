from pathlib import Path

files = {
    "app/config.py": r'''
import os

APP_NAME = os.getenv("APP_NAME", "DermaLens")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
''',

    "app/core/constants.py": r'''
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 32
JPEG_QUALITY = 95
DEFAULT_IMAGE_FORMAT = "PNG"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
''',

    "app/core/logger.py": r'''
import logging
import os

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger

logger = get_logger("dermalens")
''',

    "app/schemas/image_schema.py": r'''
from typing import Any, Optional
from pydantic import BaseModel, Field

class ImageMetrics(BaseModel):
    original_sharpness: Optional[float] = None
    enhanced_sharpness: Optional[float] = None
    original_contrast: Optional[float] = None
    enhanced_contrast: Optional[float] = None
    original_brightness: Optional[float] = None
    enhanced_brightness: Optional[float] = None
    brightness_improvement: Optional[float] = None
    quality_score: Optional[float] = None

class ImageProcessResponse(BaseModel):
    success: bool = True
    message: str = "Image processed"
    image: Optional[str] = None
    mode: str = "enhancement"
    metrics: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

class ImageUploadInfo(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size_bytes: int
''',

    "app/services/image_service.py": r'''
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS
from app.core.constants import MAX_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION
from app.core.logger import get_logger

logger = get_logger(__name__)

class ImageService:
    @staticmethod
    def decode_bytes(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("The uploaded image is empty.")

        if len(data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                width, height = image.size

                if min(width, height) < MIN_IMAGE_DIMENSION:
                    raise ValueError("Image dimensions are too small.")
                if max(width, height) > MAX_IMAGE_DIMENSION:
                    raise ValueError("Image dimensions exceed the supported limit.")

                rgb = np.array(image)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Please upload a valid JPG, PNG, or WebP image.") from exc

    @staticmethod
    def encode_base64(image: np.ndarray, image_format: str = "PNG") -> str:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Cannot encode an empty image.")

        fmt = image_format.upper()
        extension = ".jpg" if fmt in {"JPG", "JPEG"} else ".png"
        success, encoded = cv2.imencode(extension, image)

        if not success:
            raise ValueError("Image encoding failed.")

        mime = "image/jpeg" if extension == ".jpg" else "image/png"
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    @staticmethod
    def load_path(path: str | Path) -> np.ndarray:
        path = Path(path)
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported image file extension.")
        data = path.read_bytes()
        return ImageService.decode_bytes(data)

image_service = ImageService()
''',

    "app/ml/pipeline.py": r'''
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
''',

    "app/routes/process.py": r'''
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.image_service import ImageService
from app.ml.pipeline import process_image
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/process")
async def process_uploaded_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = ImageService.decode_bytes(data)
        result = process_image(image)
        result["filename"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image processing failed")
        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Check backend logs."
        ) from exc
    finally:
        await file.close()
'''
}

for filename, content in files.items():
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"SKIPPED (not empty): {path}")
        continue

    path.write_text(content.lstrip(), encoding="utf-8")
    print(f"CREATED: {path}")

print("\nDone. Existing non-empty files were preserved.")
