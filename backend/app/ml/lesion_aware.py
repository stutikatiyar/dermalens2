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

    This is a runtime reliability check.

    It is NOT Dice/IoU because ground-truth masks are not
    available for arbitrary uploaded images.
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
        connectivity=8
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

    largest_component_ratio = (
        largest_component / float(lesion_pixels)
    )

    if largest_component_ratio < MIN_LARGEST_COMPONENT_RATIO:
        return {
            "reliable": False,
            "reason": "Lesion mask is fragmented",
            "area_ratio": float(area_ratio),
            "largest_component_ratio": float(
                largest_component_ratio
            ),
        }

    return {
        "reliable": True,
        "reason": "Lesion mask passed reliability checks",
        "area_ratio": float(area_ratio),
        "largest_component_ratio": float(
            largest_component_ratio
        ),
    }


# --------------------------------------------------
# SOFT LESION MASK
# --------------------------------------------------

def create_soft_lesion_mask(
    mask,
    target_shape,
    blur_kernel=21
):
    """
    Resize and soften the lesion mask so enhancement
    transitions smoothly into surrounding skin.
    """

    if mask is None:
        raise ValueError("Lesion mask cannot be None.")

    target_height = target_shape[0]
    target_width = target_shape[1]

    resized_mask = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    resized_mask = resized_mask.astype(np.float32) / 255.0

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    soft_mask = cv2.GaussianBlur(
        resized_mask,
        (blur_kernel, blur_kernel),
        0
    )

    return np.clip(
        soft_mask,
        0.0,
        1.0
    )


# --------------------------------------------------
# LESION-AWARE CANDIDATE BLENDING
# --------------------------------------------------

def apply_lesion_aware_enhancement(
    original,
    candidate,
    lesion_mask
):
    """
    Apply an enhancement candidate primarily inside the
    segmented lesion region while preserving the surrounding
    skin from the original image.

    Handles candidates with different resolutions,
    including super-resolution outputs.
    """

    if original is None:
        raise ValueError("Original image cannot be None.")

    if candidate is None:
        raise ValueError("Candidate image cannot be None.")

    if lesion_mask is None:
        raise ValueError("Lesion mask cannot be None.")

    candidate_height, candidate_width = candidate.shape[:2]

    if original.shape[:2] != candidate.shape[:2]:

        original_resized = cv2.resize(
            original,
            (candidate_width, candidate_height),
            interpolation=cv2.INTER_LINEAR
        )

    else:

        original_resized = original

    soft_mask = create_soft_lesion_mask(
        lesion_mask,
        candidate.shape[:2]
    )

    soft_mask = soft_mask[..., np.newaxis]

    original_float = original_resized.astype(np.float32)
    candidate_float = candidate.astype(np.float32)

    result = (
        candidate_float * soft_mask
        +
        original_float * (1.0 - soft_mask)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)
