import os
import csv
import cv2
import requests
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/upload"

# Put your clear test images in this folder
INPUT_FOLDER = "benchmark_images"

# Folder where degraded test images will be saved
DEGRADED_FOLDER = "benchmark_degraded"

# Final benchmark CSV
OUTPUT_CSV = "benchmark_results.csv"


# ============================================================
# IMAGE DEGRADATION FUNCTIONS
# ============================================================

def severe_blur(image):
    """
    Simulates severe focus/motion blur.
    """
    return cv2.GaussianBlur(image, (15, 15), 0)


def low_resolution(image):
    """
    Simulates a low-resolution smartphone image by
    downscaling and then resizing back up.
    """
    height, width = image.shape[:2]

    small = cv2.resize(
        image,
        (max(1, width // 4), max(1, height // 4)),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def add_noise(image):
    """
    Adds Gaussian sensor noise.
    """
    image_float = image.astype(np.float32)

    noise = np.random.normal(
        0,
        25,
        image.shape
    ).astype(np.float32)

    noisy = image_float + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def underexposure(image):
    """
    Simulates an underexposed / poorly illuminated image.
    """
    return np.clip(
        image.astype(np.float32) * 0.35,
        0,
        255,
    ).astype(np.uint8)


def overexposure(image):
    """
    Simulates an overexposed image.
    """
    return np.clip(
        image.astype(np.float32) * 1.7,
        0,
        255,
    ).astype(np.uint8)


def jpeg_compression(image):
    """
    Simulates heavy JPEG compression.
    """
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 20],
    )

    if not success:
        return image

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )


# ============================================================
# DEGRADATION DEFINITIONS
# ============================================================

DEGRADATIONS = {
    "Severe Blur": severe_blur,
    "Low Resolution": low_resolution,
    "Noise": add_noise,
    "Underexposure": underexposure,
    "Overexposure": overexposure,
    "JPEG Compression": jpeg_compression,
}


# ============================================================
# CALL DERMALENS API
# ============================================================

def process_with_dermalens(image_path):
    """
    Sends an image to the existing DermaLens /upload endpoint
    and returns the JSON response.
    """

    with open(image_path, "rb") as image_file:

        files = {
            "file": (
                os.path.basename(image_path),
                image_file,
                "image/jpeg",
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            timeout=180,
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# SAVE DEGRADED IMAGE
# ============================================================

def save_image(image, path):
    success = cv2.imwrite(path, image)

    if not success:
        raise RuntimeError(
            f"Could not save image: {path}"
        )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    os.makedirs(
        INPUT_FOLDER,
        exist_ok=True,
    )

    os.makedirs(
        DEGRADED_FOLDER,
        exist_ok=True,
    )

    image_files = [
        file
        for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    if not image_files:

        print(
            f"\nNo images found in '{INPUT_FOLDER}'."
        )

        print(
            "Put your dermatology test images there first."
        )

        return

    results = []

    print("\n==========================================")
    print("        DERMALENS BENCHMARK")
    print("==========================================")

    print(
        f"Images found: {len(image_files)}"
    )

    print(
        f"Tests per image: {len(DEGRADATIONS)}"
    )

    print(
        f"Total tests: "
        f"{len(image_files) * len(DEGRADATIONS)}"
    )

    print("==========================================\n")


    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_index, filename in enumerate(
        image_files,
        start=1,
    ):

        input_path = os.path.join(
            INPUT_FOLDER,
            filename,
        )

        original = cv2.imread(input_path)

        if original is None:

            print(
                f"Skipping unreadable image: {filename}"
            )

            continue

        print(
            f"\n[{image_index}/{len(image_files)}] "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Apply every degradation
        # ----------------------------------------------------

        for condition, degradation_function in DEGRADATIONS.items():

            print(
                f"  → Testing {condition}..."
            )

            try:

                degraded = degradation_function(
                    original.copy()
                )

                # Create safe filename
                condition_name = (
                    condition
                    .lower()
                    .replace(" ", "_")
                )

                degraded_filename = (
                    f"{os.path.splitext(filename)[0]}"
                    f"_{condition_name}.jpg"
                )

                degraded_path = os.path.join(
                    DEGRADED_FOLDER,
                    degraded_filename,
                )

                save_image(
                    degraded,
                    degraded_path,
                )

                # ------------------------------------------------
                # Send degraded image to DermaLens
                # ------------------------------------------------

                data = process_with_dermalens(
                    degraded_path
                )

                # ------------------------------------------------
                # Extract metrics
                # ------------------------------------------------

                metrics = data.get(
                    "metrics",
                    {}
                )

                confidence = data.get(
                    "confidence",
                    {}
                )

                quality_before = metrics.get(
                    "quality_before"
                )

                quality_after = metrics.get(
                    "quality_after"
                )

                confidence_before = confidence.get(
                    "before"
                )

                confidence_after = confidence.get(
                    "after"
                )

                confidence_delta = confidence.get(
                    "delta"
                )

                accepted = data.get(
                    "enhancement_accepted"
                )

                mode = data.get(
                    "mode"
                )

                # ------------------------------------------------
                # Convert decision to readable text
                # ------------------------------------------------

                if accepted is True:
                    decision = "Accepted"

                elif accepted is False:
                    decision = "Rejected"

                else:
                    decision = "Unknown"

                # ------------------------------------------------
                # Store result
                # ------------------------------------------------

                results.append(
                    {
                        "Image": filename,
                        "Condition": condition,

                        "Quality Before": (
                            quality_before
                        ),

                        "Quality After": (
                            quality_after
                        ),

                        "Quality Improvement": (
                            quality_after
                            - quality_before
                            if quality_before is not None
                            and quality_after is not None
                            else None
                        ),

                        "AI Confidence Before": (
                            confidence_before
                        ),

                        "AI Confidence After": (
                            confidence_after
                        ),

                        "AI Confidence Delta": (
                            confidence_delta
                        ),

                        "Decision": decision,

                        "Mode": mode,

                        "Test Image": (
                            degraded_filename
                        ),
                    }
                )

                print(
                    f"     Quality: "
                    f"{quality_before:.2f}"
                    f" → "
                    f"{quality_after:.2f}"
                )

                print(
                    f"     AI Confidence: "
                    f"{confidence_before:.4f}"
                    f" → "
                    f"{confidence_after:.4f}"
                )

                print(
                    f"     Decision: "
                    f"{decision}"
                )

            except Exception as error:

                print(
                    f"     ERROR: {error}"
                )

                results.append(
                    {
                        "Image": filename,
                        "Condition": condition,
                        "Quality Before": None,
                        "Quality After": None,
                        "Quality Improvement": None,
                        "AI Confidence Before": None,
                        "AI Confidence After": None,
                        "AI Confidence Delta": None,
                        "Decision": "ERROR",
                        "Mode": None,
                        "Test Image": None,
                    }
                )


    # ========================================================
    # SAVE CSV
    # ========================================================

    if not results:

        print(
            "\nNo benchmark results generated."
        )

        return


    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)


    # ========================================================
    # SUMMARY
    # ========================================================

    valid_results = [
        result
        for result in results
        if result["Decision"] != "ERROR"
    ]

    accepted = sum(
        result["Decision"] == "Accepted"
        for result in valid_results
    )

    rejected = sum(
        result["Decision"] == "Rejected"
        for result in valid_results
    )

    print("\n==========================================")
    print("        BENCHMARK COMPLETE")
    print("==========================================")

    print(
        f"Total tests: {len(valid_results)}"
    )

    print(
        f"Accepted: {accepted}"
    )

    print(
        f"Rejected: {rejected}"
    )

    print(
        f"Results saved to: {OUTPUT_CSV}"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()