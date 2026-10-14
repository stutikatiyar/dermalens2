import sys
from pathlib import Path

import cv2

from app.ml.lesion_segmenter import segment_lesion


# --------------------------------------------------
# CHECK IMAGE PATH
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print(
        r"python test_lesion_segmentation.py "
        r'"FULL_IMAGE_PATH"'
    )
    sys.exit(1)


IMAGE_PATH = sys.argv[1]

image_path = Path(IMAGE_PATH)

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = Path("segmentation_test")

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

print("Loading image...")

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        f"Could not read image:\n{IMAGE_PATH}"
    )

print(
    f"Image loaded: "
    f"{image.shape[1]} x {image.shape[0]}"
)

print(
    f"Input: {image_path}"
)


# --------------------------------------------------
# SEGMENTATION
# --------------------------------------------------

print("\nRunning lesion segmentation...")

result = segment_lesion(
    image
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

lesion_info = result["lesion_info"]

print(
    "\n========== SEGMENTATION RESULT =========="
)

print(
    f"Lesion detected: "
    f"{lesion_info['detected']}"
)

print(
    f"Lesion area: "
    f"{lesion_info['area_pixels']} pixels"
)

print(
    f"Lesion area ratio: "
    f"{lesion_info['area_ratio']:.2%}"
)

print(
    f"Bounding box: "
    f"{lesion_info['bbox']}"
)

print(
    "=========================================="
)


# --------------------------------------------------
# UNIQUE OUTPUT NAME
# --------------------------------------------------

stem = image_path.stem

mask_path = (
    OUTPUT_DIR /
    f"{stem}_lesion_mask.png"
)

overlay_path = (
    OUTPUT_DIR /
    f"{stem}_lesion_overlay.png"
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

cv2.imwrite(
    str(mask_path),
    result["mask"]
)

cv2.imwrite(
    str(overlay_path),
    result["overlay"]
)


print(
    f"\nMask saved to:\n{mask_path}"
)

print(
    f"Overlay saved to:\n{overlay_path}"
)