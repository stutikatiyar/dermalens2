from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def load_image(source):
    """
    Load an image from bytes, a file path, or a file-like object.
    Returns an RGB NumPy array.
    """
    try:
        if isinstance(source, (str, Path)):
            if Path(source).stat().st_size > MAX_FILE_SIZE:
                raise ValueError("Image exceeds the 10 MB limit.")
            image = Image.open(source)

        elif isinstance(source, bytes):
            if len(source) > MAX_FILE_SIZE:
                raise ValueError("Image exceeds the 10 MB limit.")
            image = Image.open(BytesIO(source))

        elif hasattr(source, "read"):
            data = source.read(MAX_FILE_SIZE + 1)
            if len(data) > MAX_FILE_SIZE:
                raise ValueError("Image exceeds the 10 MB limit.")
            image = Image.open(BytesIO(data))

        else:
            raise TypeError("Provide image bytes, a file path, or a file-like object.")

        if image.format not in ALLOWED_FORMATS:
            raise ValueError("Only JPEG, PNG and WEBP images are supported.")

        image.verify()

        if isinstance(source, (str, Path)):
            image = Image.open(source)
        elif isinstance(source, bytes):
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(BytesIO(data))

        image = image.convert("RGB")

        if image.width < 32 or image.height < 32:
            raise ValueError("Image dimensions must be at least 32 x 32.")

        if image.width > 8000 or image.height > 8000:
            raise ValueError("Image dimensions exceed the 8000-pixel limit.")

        return np.array(image)

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc