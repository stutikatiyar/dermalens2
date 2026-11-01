
import cv2
import json
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DEBUG_DIR = (
    Path(__file__).resolve().parents[2]
    / "debug"
    / "lesion_crop"
)

ORIGINAL_PATH = DEBUG_DIR / "mprnet_lesion_crop_original_crop.png"
MASK_PATH = DEBUG_DIR / "mprnet_lesion_crop_mask.png"

OUTPUT_DIR = DEBUG_DIR / "classical_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# IMAGE QUALITY METRICS
# --------------------------------------------------

def calculate_metrics(image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        if mask.shape != gray.shape:
            mask = cv2.resize(
                mask,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        region = mask >= 128
    else:
        region = np.ones(gray.shape, dtype=bool)

    if not np.any(region):
        region = np.ones(gray.shape, dtype=bool)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 20, 80)

    return {
        "sharpness_laplacian_variance": float(
            laplacian[region].var()
        ),
        "edge_pixel_percentage": float(
            100.0 * np.mean(edges[region] > 0)
        ),
        "contrast_standard_deviation": float(
            gray[region].std()
        ),
        "mean_brightness": float(
            gray[region].mean()
        ),
    }


# --------------------------------------------------
# ENHANCEMENT METHODS
# --------------------------------------------------

def apply_clahe(image):
    """Apply mild local contrast enhancement in LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)

    result = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def apply_gentle_denoising(image):
    """Reduce mild color noise without aggressive smoothing."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def apply_conservative_sharpening(image):
    """Apply a low-strength unsharp mask."""
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
    )

    result = cv2.addWeighted(
        image,
        1.20,
        blurred,
        -0.20,
        0,
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe_then_denoise(image):
    """Test contrast enhancement followed by gentle denoising."""
    return apply_gentle_denoising(apply_clahe(image))


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(
            f"Original crop not found: {ORIGINAL_PATH}\n"
            "Run the lesion-crop experiment first."
        )

    original = cv2.imread(str(ORIGINAL_PATH))

    if original is None:
        raise RuntimeError("Could not read the original crop.")

    mask = None

    if MASK_PATH.exists():
        mask = cv2.imread(
            str(MASK_PATH),
            cv2.IMREAD_GRAYSCALE,
        )

    candidates = {
        "original": original,
        "clahe": apply_clahe(original),
        "gentle_denoising": apply_gentle_denoising(original),
        "conservative_sharpening": apply_conservative_sharpening(
            original
        ),
        "clahe_then_denoise": apply_clahe_then_denoise(original),
    }

    results = {}

    for name, image in candidates.items():
        output_path = OUTPUT_DIR / f"{name}.png"

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {output_path}")

        results[name] = {
            "metrics": calculate_metrics(image, mask),
            "saved_image": str(output_path),
        }

        print(f"\n{name}")
        print(json.dumps(results[name]["metrics"], indent=2))

    # Create a visual comparison image.
    # Each candidate is resized to the same display size.
    display_images = []

    for name, image in candidates.items():
        preview = cv2.resize(
            image,
            (image.shape[1] * 2, image.shape[0] * 2),
            interpolation=cv2.INTER_NEAREST,
        )

        label_height = 35
        labeled = np.zeros(
            (
                preview.shape[0] + label_height,
                preview.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        labeled[label_height:] = preview

        cv2.putText(
            labeled,
            name,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        display_images.append(labeled)

    # Arrange candidates in two columns.
    columns = 2
    rows = []

    for i in range(0, len(display_images), columns):
        row_images = display_images[i:i + columns]

        if len(row_images) < columns:
            row_images.append(
                np.zeros_like(row_images[0])
            )

        row_height = max(img.shape[0] for img in row_images)
        normalized = []

        for img in row_images:
            if img.shape[0] < row_height:
                padding = np.zeros(
                    (
                        row_height - img.shape[0],
                        img.shape[1],
                        3,
                    ),
                    dtype=np.uint8,
                )
                img = np.vstack([img, padding])

            normalized.append(img)

        rows.append(np.hstack(normalized))

    comparison_image = np.vstack(rows)
    comparison_path = OUTPUT_DIR / "comparison_grid.png"

    cv2.imwrite(str(comparison_path), comparison_image)

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n----------------------------------")
    print("EXPERIMENT COMPLETE")
    print("----------------------------------")
    print(f"Comparison image: {comparison_path}")
    print(f"Metrics JSON:     {metrics_path}")
    print(f"Output folder:    {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
