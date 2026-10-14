
import os
import csv
import base64

import cv2
import numpy as np
import requests

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

API_URL = "http://127.0.0.1:8000/upload"

CLEAN_DIR = "benchmark_images"
DEGRADED_DIR = "benchmark_degraded"
OUTPUT_CSV = os.path.join(
    "evaluation_results",
    "enhancement_safety_results.csv",
)

DEGRADATION_SUFFIXES = [
    "_severe_blur",
    "_low_resolution",
    "_noise",
    "_underexposure",
    "_overexposure",
    "_jpeg_compression",
]


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def calculate_psnr(reference, image):
    if reference.shape != image.shape:
        return None

    value = peak_signal_noise_ratio(
        reference, image, data_range=255
    )

    return float(value) if np.isfinite(value) else 100.0


def calculate_ssim(reference, image):
    if reference.shape != image.shape:
        return None

    return float(
        structural_similarity(
            reference,
            image,
            channel_axis=-1,
            data_range=255,
        )
    )


def send_to_dermalens(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    os.path.basename(image_path),
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=180,
        )

    response.raise_for_status()
    return response.json()


def decode_api_image(data):
    encoded = data.get("image")

    if not encoded:
        raise ValueError("API response has no 'image' field")

    raw = base64.b64decode(encoded)
    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("Could not decode enhanced image")

    return image


def main():
    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True,
    )

    degraded_files = sorted(
        name for name in os.listdir(DEGRADED_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    results = []

    print(f"Found {len(degraded_files)} degraded images.")
    print("Sending each image to the existing DermaLens API.")
    print("Existing benchmark CSV files will not be modified.")

    for index, filename in enumerate(degraded_files, start=1):
        stem = os.path.splitext(filename)[0]

        clean_stem = stem
        degradation = "unknown"

        for suffix in DEGRADATION_SUFFIXES:
            if stem.endswith(suffix):
                clean_stem = stem[:-len(suffix)]
                degradation = suffix.lstrip("_")
                break

        clean_path = None
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = os.path.join(
                CLEAN_DIR, clean_stem + extension
            )
            if os.path.isfile(candidate):
                clean_path = candidate
                break

        if clean_path is None:
            print(f"[{index}] Skipped: no clean reference for {filename}")
            continue

        degraded_path = os.path.join(DEGRADED_DIR, filename)

        try:
            clean = cv2.imread(clean_path)
            degraded = cv2.imread(degraded_path)

            if clean is None or degraded is None:
                raise ValueError("Could not read clean or degraded image")

            data = send_to_dermalens(degraded_path)
            enhanced = decode_api_image(data)

            if clean.shape != enhanced.shape:
                enhanced = cv2.resize(
                    enhanced,
                    (clean.shape[1], clean.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            if degraded.shape != enhanced.shape:
                degraded_for_comparison = cv2.resize(
                    degraded,
                    (enhanced.shape[1], enhanced.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                degraded_for_comparison = degraded

            metrics = data.get("metrics", {})

            row = {
                "image": clean_stem,
                "degradation": degradation,
                "input_file": filename,
                "enhancement_mode": data.get("mode"),
                "enhancement_accepted": data.get(
                    "enhancement_accepted",
                    data.get("accepted"),
                ),
                "mask_reliable": data.get("mask_reliable"),
                "area_ratio": data.get("area_ratio"),
                "psnr_degraded_vs_clean": calculate_psnr(
                    clean, degraded
                ) if clean.shape == degraded.shape else None,
                "ssim_degraded_vs_clean": calculate_ssim(
                    clean, degraded
                ) if clean.shape == degraded.shape else None,
                "psnr_enhanced_vs_clean": calculate_psnr(
                    clean, enhanced
                ),
                "ssim_enhanced_vs_clean": calculate_ssim(
                    clean, enhanced
                ),
                "psnr_enhanced_vs_input": calculate_psnr(
                    degraded_for_comparison, enhanced
                ),
                "ssim_enhanced_vs_input": calculate_ssim(
                    degraded_for_comparison, enhanced
                ),
                "clean_sharpness": calculate_sharpness(clean),
                "enhanced_sharpness": calculate_sharpness(enhanced),
                "clean_contrast": calculate_contrast(clean),
                "enhanced_contrast": calculate_contrast(enhanced),
                "clean_brightness": calculate_brightness(clean),
                "enhanced_brightness": calculate_brightness(enhanced),
                "api_quality_before": metrics.get("quality_before"),
                "api_quality_after": metrics.get("quality_after"),
                "confidence_before": data.get("confidence", {}).get("before"),
                "confidence_after": data.get("confidence", {}).get("after"),
            }

            results.append(row)

            print(
                f"[{index}/{len(degraded_files)}] {filename}: "
                f"SSIM clean-reference "
                f"{row['ssim_degraded_vs_clean']} -> "
                f"{row['ssim_enhanced_vs_clean']}; "
                f"mode={row['enhancement_mode']}"
            )

        except Exception as error:
            print(f"[{index}] ERROR for {filename}: {error}")

    if not results:
        print("No results generated. Check the folders and API.")
        return

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted {len(results)} tests.")
    print(f"New results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
