"""
Lesion-crop restoration experiment for DermaLens.

This is an experimental comparison helper. It does not alter the production
selection logic. It returns the raw crop restoration and a blended full image
so both can be inspected independently.

Expected inputs:
- original_bgr: uint8 OpenCV BGR image
- lesion_mask: 2D segmentation mask, values in [0,1] or [0,255]
- restore_fn: callable(image_bgr) -> (restored_bgr, metadata), e.g.
  enhance_mprnet_bgr from mprnet_adapter.py
"""
from pathlib import Path
import cv2
import numpy as np


def _normalize_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("lesion_mask must be a non-empty 2D array")
    mask = mask.astype(np.float32)
    if not np.all(np.isfinite(mask)):
        raise ValueError("lesion_mask contains non-finite values")
    low, high = float(mask.min()), float(mask.max())
    if low >= 0 and high <= 1:
        return mask
    if low >= 0 and high <= 255:
        return mask / 255.0
    raise ValueError("lesion_mask values must be in [0,1] or [0,255]")


def enhance_lesion_crop(
    original_bgr,
    lesion_mask,
    restore_fn,
    margin_ratio=1.0,
    min_crop_size=96,
    blend_strength=0.85,
):
    """
    Returns a dictionary:
      bbox: crop bounds in original image coordinates
      crop_original: padded original crop
      crop_restored: raw restored crop (resized to crop dimensions if needed)
      crop_mask: corresponding soft crop mask
      blended_full: full-size output with restored crop blended in
      metadata: restoration metadata

    `blend_strength` controls how much restored crop is applied inside the mask.
    It does not claim restored pixels are medically accurate.
    """
    if not isinstance(original_bgr, np.ndarray) or original_bgr.ndim != 3:
        raise ValueError("original_bgr must be a valid 3-channel image")
    if original_bgr.dtype != np.uint8 or original_bgr.shape[2] != 3:
        raise ValueError("original_bgr must be uint8 BGR")
    if restore_fn is None or not callable(restore_fn):
        raise ValueError("restore_fn must be a callable restoration function")

    height, width = original_bgr.shape[:2]
    mask = _normalize_mask(lesion_mask)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (mask >= 0.5).astype(np.uint8)

    if int(binary.sum()) == 0:
        raise ValueError("Lesion mask contains no pixels at threshold 0.5")

    ys, xs = np.where(binary > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    lesion_w, lesion_h = x1 - x0, y1 - y0

    # Add contextual skin margin around the segmented lesion.
    margin = int(round(max(lesion_w, lesion_h) * max(0.0, margin_ratio)))
    cx0 = max(0, x0 - margin)
    cy0 = max(0, y0 - margin)
    cx1 = min(width, x1 + margin)
    cy1 = min(height, y1 + margin)

    # If the ROI is tiny, expand it around its centre while staying in bounds.
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    if min(crop_w, crop_h) < min_crop_size:
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        target_w = min(width, max(crop_w, min_crop_size))
        target_h = min(height, max(crop_h, min_crop_size))
        cx0 = max(0, center_x - target_w // 2)
        cy0 = max(0, center_y - target_h // 2)
        cx1 = min(width, cx0 + target_w)
        cy1 = min(height, cy0 + target_h)
        cx0 = max(0, cx1 - target_w)
        cy0 = max(0, cy1 - target_h)

    crop = original_bgr[cy0:cy1, cx0:cx1].copy()
    crop_mask_hard = mask[cy0:cy1, cx0:cx1]
    crop_mask = cv2.GaussianBlur(crop_mask_hard, (0, 0), sigmaX=2.0)
    crop_mask = np.clip(crop_mask, 0.0, 1.0)

    restored, metadata = restore_fn(crop)
    if not isinstance(restored, np.ndarray) or restored.ndim != 3 or restored.shape[2] != 3:
        raise ValueError("restore_fn returned an invalid image")
    if restored.dtype != np.uint8:
        restored = np.clip(restored, 0, 255).astype(np.uint8)

    # This adapter currently returns same-resolution MPRNet output. Keep this
    # helper robust if a future restoration function changes dimensions.
    crop_h, crop_w = crop.shape[:2]
    if restored.shape[:2] != (crop_h, crop_w):
        restored = cv2.resize(
            restored, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC
        )

    # Keep the raw crop available for the dashboard, but blend only inside
    # a feathered mask and use a tunable strength for visual experiments.
    alpha = (crop_mask * float(np.clip(blend_strength, 0.0, 1.0)))[..., None]
    original_float = crop.astype(np.float32)
    restored_float = restored.astype(np.float32)
    blended_crop = np.clip(
        restored_float * alpha + original_float * (1.0 - alpha), 0, 255
    ).astype(np.uint8)

    blended_full = original_bgr.copy()
    blended_full[cy0:cy1, cx0:cx1] = blended_crop

    return {
        "bbox": {"x": cx0, "y": cy0, "width": cx1 - cx0, "height": cy1 - cy0},
        "lesion_bbox": {"x": x0, "y": y0, "width": lesion_w, "height": lesion_h},
        "crop_original": crop,
        "crop_restored": restored,
        "crop_mask": np.clip(crop_mask * 255.0, 0, 255).astype(np.uint8),
        "blended_full": blended_full,
        "metadata": metadata if isinstance(metadata, dict) else {"metadata": str(metadata)},
    }


def save_lesion_crop_debug(result, output_dir, prefix="mprnet_lesion_crop"):
    """Save diagnostic images and a JSON metadata file for local inspection."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for key, suffix in (
        ("crop_original", "original_crop.png"),
        ("crop_restored", "restored_crop_raw.png"),
        ("crop_mask", "mask.png"),
        ("blended_full", "blended_full.png"),
    ):
        image = result.get(key)
        if image is None:
            continue
        path = output_dir / f"{prefix}_{suffix}"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to save debug image: {path}")
        paths[key] = str(path)

    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bbox": result.get("bbox"),
                "lesion_bbox": result.get("lesion_bbox"),
                "metadata": result.get("metadata", {}),
                "saved_images": paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path)
    return paths
