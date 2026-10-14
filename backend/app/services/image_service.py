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
