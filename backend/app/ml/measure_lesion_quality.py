
import cv2
import numpy as np
import json
from pathlib import Path

# Debug images created by your lesion-crop experiment
DEBUG_DIR = (
    Path(__file__).resolve().parents[2]
    / "debug"
    / "lesion_crop"
)

ORIGINAL_PATH = DEBUG_DIR / "mprnet_lesion_crop_original_crop.png"
RESTORED_PATH = DEBUG_DIR / "mprnet_lesion_crop_restored_crop_raw.png"
MASK_PATH = DEBUG_DIR / "mprnet_lesion_crop_mask.png"


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_strength(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean())


def main():
    for path in [ORIGINAL_PATH, RESTORED_PATH, MASK_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run the lesion-crop experiment first."
            )

    original = cv2.imread(str(ORIGINAL_PATH))
    restored = cv2.imread(str(RESTORED_PATH))
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if original is None or restored is None or mask is None:
        raise RuntimeError("Could not read one or more debug images.")

    if original.shape != restored.shape:
        raise ValueError(
            f"Image sizes differ: {original.shape} vs {restored.shape}"
        )

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Use the central white lesion region of the mask.
    lesion = mask >= 128

    if not np.any(lesion):
        raise ValueError("The lesion mask is empty.")

    # Also measure the whole crop, for comparison.
    results = {}

    for name, image in [
        ("original", original),
        ("mprnet_restored", restored),
    ]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        results[name] = {
            "whole_crop_sharpness": sharpness(image),
            "whole_crop_edge_strength": edge_strength(image),
            "lesion_sharpness": float(
                laplacian[lesion].var()
            ),
            "lesion_mean_brightness": float(
                gray[lesion].mean()
            ),
        }

    results["comparison"] = {
        "lesion_pixel_difference_mean": float(
            np.abs(
                original.astype(np.float32)
                - restored.astype(np.float32)
            )[lesion].mean()
        ),
        "lesion_sharpness_change_percent": (
            (
                results["mprnet_restored"]["lesion_sharpness"]
                - results["original"]["lesion_sharpness"]
            )
            / max(results["original"]["lesion_sharpness"], 1e-9)
        ) * 100,
    }

    output_path = DEBUG_DIR / "quality_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
