import cv2
import numpy as np


# --------------------------------------------------
# LESION MASK RELIABILITY
# --------------------------------------------------

MIN_LESION_AREA_RATIO = 0.005
MAX_LESION_AREA_RATIO = 0.60
MIN_LARGEST_COMPONENT_RATIO = 0.60


def validate_lesion_mask(mask):
    """
    Check whether a predicted lesion mask is reliable enough
    to be used for lesion-aware enhancement.

    This is a runtime reliability check. It is NOT Dice/IoU,
    because ground-truth masks are not available for arbitrary
    uploaded images.
    """
    if mask is None:
        return {
            "reliable": False,
            "reason": "No lesion mask generated",
            "area_ratio": 0.0,
            "largest_component_ratio": 0.0,
        }

    if not isinstance(mask, np.ndarray):
        return {
            "reliable": False,
            "reason": "Invalid lesion mask type",
            "area_ratio": 0.0,
            "largest_component_ratio": 0.0,
        }

    if mask.ndim != 2 or mask.size == 0:
        return {
            "reliable": False,
            "reason": "Invalid lesion mask dimensions",
            "area_ratio": 0.0,
            "largest_component_ratio": 0.0,
        }

    binary = mask > 0
    total_pixels = binary.size
    lesion_pixels = int(np.count_nonzero(binary))

    if lesion_pixels == 0:
        return {
            "reliable": False,
            "reason": "No lesion detected",
            "area_ratio": 0.0,
            "largest_component_ratio": 0.0,
        }

    area_ratio = lesion_pixels / float(total_pixels)

    if area_ratio < MIN_LESION_AREA_RATIO:
        return {
            "reliable": False,
            "reason": "Detected lesion region is too small",
            "area_ratio": float(area_ratio),
            "largest_component_ratio": 0.0,
        }

    if area_ratio > MAX_LESION_AREA_RATIO:
        return {
            "reliable": False,
            "reason": "Detected lesion region is excessively large",
            "area_ratio": float(area_ratio),
            "largest_component_ratio": 0.0,
        }

    binary_uint8 = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_uint8,
        connectivity=8,
    )

    if num_labels <= 1:
        return {
            "reliable": False,
            "reason": "No connected lesion component found",
            "area_ratio": float(area_ratio),
            "largest_component_ratio": 0.0,
        }

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component = int(np.max(component_areas))
    largest_component_ratio = largest_component / float(lesion_pixels)

    if largest_component_ratio < MIN_LARGEST_COMPONENT_RATIO:
        return {
            "reliable": False,
            "reason": "Lesion mask is fragmented",
            "area_ratio": float(area_ratio),
            "largest_component_ratio": float(largest_component_ratio),
        }

    return {
        "reliable": True,
        "reason": "Lesion mask passed reliability checks",
        "area_ratio": float(area_ratio),
        "largest_component_ratio": float(largest_component_ratio),
    }


# --------------------------------------------------
# SOFT LESION MASK
# --------------------------------------------------

def create_soft_lesion_mask(mask, target_shape, blur_kernel=21):
    """Resize and soften a lesion mask with safe normalization."""
    if mask is None:
        raise ValueError("Lesion mask cannot be None.")

    if not isinstance(mask, np.ndarray) or mask.size == 0:
        raise ValueError("Lesion mask must be a non-empty NumPy array.")

    if mask.ndim != 2:
        raise ValueError("Lesion mask must be a 2D array.")

    target_height, target_width = target_shape[:2]
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target shape must have positive dimensions.")

    mask_float = mask.astype(np.float32)

    if not np.all(np.isfinite(mask_float)):
        raise ValueError("Lesion mask contains non-finite values.")

    mask_min = float(mask_float.min())
    mask_max = float(mask_float.max())

    if mask_min >= 0.0 and mask_max <= 1.0:
        normalized_mask = mask_float
    elif mask_min >= 0.0 and mask_max <= 255.0:
        normalized_mask = mask_float / 255.0
    else:
        raise ValueError("Mask values must be in [0, 1] or [0, 255].")

    resized_mask = cv2.resize(
        normalized_mask,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    blur_kernel = max(1, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    if blur_kernel > 1:
        soft_mask = cv2.GaussianBlur(
            resized_mask,
            (blur_kernel, blur_kernel),
            0,
        )
    else:
        soft_mask = resized_mask

    return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------
# LESION-AWARE CANDIDATE BLENDING
# --------------------------------------------------

def apply_lesion_aware_enhancement(original, candidate, lesion_mask):
    """
    Apply an enhancement candidate primarily inside the segmented
    lesion region while preserving surrounding skin from the original.

    Supports candidates with different resolutions, including
    super-resolution outputs. Images are expected to be uint8,
    three-channel OpenCV images in BGR format.
    """
    if original is None:
        raise ValueError("Original image cannot be None.")
    if candidate is None:
        raise ValueError("Candidate image cannot be None.")
    if lesion_mask is None:
        raise ValueError("Lesion mask cannot be None.")

    for name, image in (("original", original), ("candidate", candidate)):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be a non-empty 3-channel image.")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must have dtype uint8.")

    candidate_height, candidate_width = candidate.shape[:2]
    original_height, original_width = original.shape[:2]

    # A super-resolution candidate has more pixels than the original.
    # Preserve that resolution instead of blending it with an upscaled
    # original, which can erase most of the super-resolution effect.
    is_higher_resolution = (
        candidate_height > original_height
        or candidate_width > original_width
    )

    if is_higher_resolution:
        soft_mask = create_soft_lesion_mask(
            lesion_mask,
            candidate.shape[:2],
            blur_kernel=max(3, int(round(21 * candidate_width / max(original_width, 1))) | 1),
        )[..., np.newaxis]

        original_resized = cv2.resize(
            original,
            (candidate_width, candidate_height),
            interpolation=cv2.INTER_CUBIC,
        )
        original_float = original_resized.astype(np.float32)
        candidate_float = candidate.astype(np.float32)

        # Keep the candidate across the whole image so its 2x resolution
        # survives. Use the lesion mask only as a gentle local emphasis;
        # don't replace the rest of the image with an interpolated original.
        lesion_weight = 0.20 * soft_mask
        result = candidate_float * (1.0 - lesion_weight) + (
            candidate_float * soft_mask + original_float * (1.0 - soft_mask)
        ) * lesion_weight

        print(
            "LESION-AWARE SR:",
            f"{original_width}x{original_height} -> {candidate_width}x{candidate_height}",
            "| output resolution preserved",
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    # Same-resolution candidates: preserve the original conservative behavior.
    soft_mask = create_soft_lesion_mask(
        lesion_mask,
        candidate.shape[:2],
    )[..., np.newaxis]

    original_float = original.astype(np.float32)
    candidate_float = candidate.astype(np.float32)

    print(
        "PRE-BLEND DIFFERENCE:",
        float(np.mean(np.abs(candidate_float - original_float))),
    )
    print(
        "MASK COVERAGE:",
        float(np.mean(soft_mask)),
    )

    result = candidate_float * soft_mask + original_float * (1.0 - soft_mask)
    return np.clip(result, 0, 255).astype(np.uint8)
