import os
import csv
import time
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

# Timestamped benchmark CSV to prevent overwriting previous experiment runs
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"benchmark_results_{TIMESTAMP}.csv"


# ============================================================
# IMAGE DEGRADATION FUNCTIONS
# ============================================================

def severe_blur(image):
    """
    Simulates severe focus/motion blur using Gaussian blur kernel.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Blurred image array.
    """
    kernel_size = (15, 15)
    sigma_x = 0
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def low_resolution(image):
    """
    Simulates a low-resolution smartphone image by
    downscaling and then resizing back up to original dimensions.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Resolution-degraded image array.
    """
    height, width = image.shape[:2]

    small_width = max(1, width // 4)
    small_height = max(1, height // 4)

    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def add_noise(image):
    """
    Adds Gaussian sensor noise to simulate poor sensor quality or high ISO.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Noisy image array.
    """
    image_float = image.astype(np.float32)

    mean = 0
    sigma = 25
    noise = np.random.normal(
        mean,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image_float + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def underexposure(image):
    """
    Simulates an underexposed / poorly illuminated dermatological image.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Darkness-degraded image array.
    """
    scaling_factor = 0.35
    
    return np.clip(
        image.astype(np.float32) * scaling_factor,
        0,
        255,
    ).astype(np.uint8)


def overexposure(image):
    """
    Simulates an overexposed or flash-blown image.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Overexposed image array.
    """
    scaling_factor = 1.7
    
    return np.clip(
        image.astype(np.float32) * scaling_factor,
        0,
        255,
    ).astype(np.uint8)


def jpeg_compression(image):
    """
    Simulates heavy JPEG compression artifacts.
    
    Parameters:
        image (numpy.ndarray): Input BGR image array.
        
    Returns:
        numpy.ndarray: Compressed image array.
    """
    quality_setting = 20
    
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality_setting],
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
    and returns the complete JSON response payload.
    
    Parameters:
        image_path (str): File system path to test image.
        
    Returns:
        dict: Parsed JSON response dictionary.
    """
    timeout_limit = 600

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
            timeout=timeout_limit,
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# SAVE DEGRADED IMAGE
# ============================================================

def save_image(image, path):
    """
    Saves an image array safely to disk.
    
    Parameters:
        image (numpy.ndarray): Image array to write.
        path (str): Target output file path.
    """
    success = cv2.imwrite(path, image)

    if not success:
        raise RuntimeError(
            f"Could not save image: {path}"
        )


# ============================================================
# FORMATTING UTILITIES FOR SAFE LOGGING
# ============================================================

def format_float_safe(value, precision=2, default="N/A"):
    """
    Safely formats float values for print output without triggering NoneType crashes.
    
    Parameters:
        value (float/None): Numerical value to format.
        precision (int): Decimal places to display.
        default (str): Fallback text when value is None.
        
    Returns:
        str: Formatted numeric string or default representation.
    """
    if value is None:
        return default
    return f"{value:.{precision}f}"


# ============================================================
# MAIN BENCHMARK EXECUTION
# ============================================================

def main():
    """
    Main benchmark execution runner. Iterates through input test images,
    applies synthetic degradation conditions, triggers API evaluations,
    extracts performance metrics, and logs complete experiment CSV records.
    """

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
    print("        DERMALENS BENCHMARK RUNNER        ")
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
    # Process every image in input directory
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
            f"Processing Source Image: {filename}"
        )

        # ----------------------------------------------------
        # Apply every degradation condition systematically
        # ----------------------------------------------------

        for condition, degradation_function in DEGRADATIONS.items():

            print(
                f"  → Testing Condition: {condition}..."
            )

            try:

                degraded = degradation_function(
                    original.copy()
                )

                # Create safe filename representing condition
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
                # Send degraded image to DermaLens API
                # ------------------------------------------------

                data = process_with_dermalens(
                    degraded_path
                )

                # ------------------------------------------------
                # Extract quality metrics directly from response
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

                # Extract exact relative quality improvement from API response
                quality_improvement = metrics.get(
                    "quality_improvement"
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
                # Convert acceptance decision to readable string
                # ------------------------------------------------

                if accepted is True:
                    decision = "Accepted"

                elif accepted is False:
                    decision = "Rejected"

                else:
                    decision = "Unknown"

                # ------------------------------------------------
                # Store full experimental record
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
                            quality_improvement
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

                # Print progress with safe formatting helpers
                q_b_str = format_float_safe(quality_before, 2)
                q_a_str = format_float_safe(quality_after, 2)
                c_b_str = format_float_safe(confidence_before, 4)
                c_a_str = format_float_safe(confidence_after, 4)

                print(
                    f"     Quality Score: "
                    f"{q_b_str}"
                    f" → "
                    f"{q_a_str}"
                )

                print(
                    f"     AI Confidence: "
                    f"{c_b_str}"
                    f" → "
                    f"{c_a_str}"
                )

                print(
                    f"     Decision Status: "
                    f"{decision}"
                )

            except Exception as error:

                print(
                    f"     ERROR encountered during processing: {error}"
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
    # SAVE BENCHMARK CSV RECORD
    # ========================================================

    if not results:

        print(
            "\nNo benchmark results were generated during this run."
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
    # SUMMARY ANALYSIS REPORTING
    # ========================================================

    valid_results = [
        result
        for result in results
        if result["Decision"] != "ERROR"
    ]

    error_count = sum(
        result["Decision"] == "ERROR"
        for result in results
    )

    accepted_count = sum(
        result["Decision"] == "Accepted"
        for result in valid_results
    )

    rejected_count = sum(
        result["Decision"] == "Rejected"
        for result in valid_results
    )

    print("\n==========================================")
    print("        BENCHMARK RUN SUMMARY COMPLETE     ")
    print("==========================================")

    print(
        f"Total Tests Executed: {len(results)}"
    )

    print(
        f"Valid Tests Processed: {len(valid_results)}"
    )

    print(
        f"Enhancements Accepted: {accepted_count}"
    )

    print(
        f"Enhancements Rejected: {rejected_count}"
    )

    print(
        f"Errors Encounted: {error_count}"
    )

    print(
        f"Experiment output recorded to: {OUTPUT_CSV}"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()