
import cv2
import json
import numpy as np
from pathlib import Path

from app.ml.lesion_segmenter import segment_lesion


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEBUG_DIR = BACKEND_DIR / "debug" / "lesion_crop"
OUTPUT_DIR = DEBUG_DIR / "lesion_only_clahe_fresh_mask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_PATH = DEBUG_DIR / "original_full.png"

CLAHE_CLIP_LIMIT = 1.2
BLEND_STRENGTH = 0.35
MASK_BLUR_SIZE = 21


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def create_soft_mask(binary_mask):
    mask = binary_mask.astype(np.float32) / 255.0

    # Ensure odd blur kernel.
    kernel = MASK_BLUR_SIZE
    if kernel % 2 == 0:
        kernel += 1

    return np.clip(
        cv2.GaussianBlur(mask, (kernel, kernel), 0),
        0.0,
        1.0,
    )


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l)
    enhanced = cv2.cvtColor(
        cv2.merge([enhanced_l, a, b]),
        cv2.COLOR_LAB2BGR,
    )
    return enhanced


def blend_lesion_only(original, enhanced, soft_mask):
    alpha = (soft_mask * BLEND_STRENGTH)[:, :, None]

    result = (
        original.astype(np.float32) * (1.0 - alpha)
        + enhanced.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def calculate_metrics(original, result, binary_mask):
    diff = np.abs(
        result.astype(np.float32) - original.astype(np.float32)
    ).mean(axis=2)

    lesion = binary_mask > 0
    outside = ~lesion

    def region_stats(region):
        if not np.any(region):
            return {
                "pixel_count": 0,
                "mean_absolute_change": None,
                "percent_pixels_changed": None,
            }

        values = diff[region]
        changed = diff > 2.0

        return {
            "pixel_count": int(region.sum()),
            "mean_absolute_change": float(values.mean()),
            "percent_pixels_changed": float(
                100.0 * np.mean(changed[region])
            ),
        }

    return {
        "lesion_region": region_stats(lesion),
        "outside_binary_mask": region_stats(outside),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    original = read_image(ORIGINAL_PATH)
    height, width = original.shape[:2]

    print(f"Original image dimensions: {width}x{height}")
    print("Generating a fresh segmentation mask...")

    segmentation = segment_lesion(original)
    binary_mask = segmentation["mask"]

    # Validate dimensions; never stretch a crop mask over the full image.
    if binary_mask.shape != (height, width):
        raise ValueError(
            f"Mask shape {binary_mask.shape} does not match "
            f"image shape {(height, width)}"
        )

    lesion_info = segmentation["lesion_info"]

    print("Segmentation information:")
    print(json.dumps(lesion_info, indent=2))

    if not lesion_info.get("detected", False):
        raise RuntimeError(
            "No lesion detected. Inspect the segmentation output "
            "before attempting enhancement."
        )

    soft_mask = create_soft_mask(binary_mask)

    enhanced = apply_clahe(original)
    result = blend_lesion_only(original, enhanced, soft_mask)

    # Difference map: brightness represents magnitude of change.
    difference = np.abs(
        result.astype(np.float32) - original.astype(np.float32)
    ).mean(axis=2)

    difference_visual = np.clip(
        difference * 8.0, 0, 255
    ).astype(np.uint8)

    difference_color = cv2.applyColorMap(
        difference_visual,
        cv2.COLORMAP_JET,
    )
    difference_color[difference <= 1.0] = (0, 0, 0)

    # Draw the binary segmentation contour for inspection.
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boundary_preview = result.copy()
    cv2.drawContours(
        boundary_preview,
        contours,
        -1,
        (0, 255, 0),
        1,
    )

    outputs = {
        "fresh_binary_mask.png": binary_mask,
        "fresh_soft_mask.png": (soft_mask * 255).astype(np.uint8),
        "original.png": original,
        "enhanced_lesion_only.png": result,
        "difference_map.png": difference_color,
        "mask_boundary_preview.png": boundary_preview,
        "full_image_clahe_reference.png": enhanced,
    }

    for filename, image in outputs.items():
        path = OUTPUT_DIR / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save {path}")

    metrics = {
        "image_dimensions": {
            "width": width,
            "height": height,
        },
        "mask_dimensions": {
            "width": int(binary_mask.shape[1]),
            "height": int(binary_mask.shape[0]),
        },
        "segmentation_info": lesion_info,
        "settings": {
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "blend_strength": BLEND_STRENGTH,
            "mask_blur_size": MASK_BLUR_SIZE,
        },
        "pixel_change": calculate_metrics(
            original, result, binary_mask
        ),
        "note": (
            "Pixel-change measurements validate localization only. "
            "They do not prove diagnostic accuracy or recovery of "
            "missing image detail."
        ),
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nFRESH-MASK VALIDATION COMPLETE")
    print(f"Output folder: {OUTPUT_DIR}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
