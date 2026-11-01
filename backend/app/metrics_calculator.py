import math
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def sanitize_float(val: float, default: float = 0.0) -> float:
    """Replaces inf/nan floats with JSON-compliant safe floating numbers."""
    if math.isinf(val) or math.isnan(val) or np.isinf(val) or np.isnan(val):
        return default
    return float(val)

def compute_laplacian_variance(image_np: np.ndarray) -> float:
    """Calculates Laplacian variance to detect image blur and sharpness."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) safely."""
    try:
        # If images are identical, skimage returns inf
        score = psnr(original, processed, data_range=255)
        if math.isinf(score) or np.isinf(score):
            return 100.0  # Perfect reconstruction cap
        return sanitize_float(score)
    except Exception:
        return 0.0

def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) safely."""
    try:
        if len(original.shape) == 3:
            score, _ = ssim(original, processed, channel_axis=2, full=True, data_range=255)
        else:
            score, _ = ssim(original, processed, full=True, data_range=255)
        return sanitize_float(score, default=1.0)
    except Exception:
        return 0.0

def compute_rmse(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculates Root Mean Square Error (RMSE) safely."""
    diff = original.astype(np.float64) - processed.astype(np.float64)
    rmse_val = np.sqrt(np.mean(diff ** 2))
    return sanitize_float(rmse_val)

def compute_color_shift_delta_e(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> float:
    """
    Computes mean color distance (ΔE) in CIELAB color space 
    to measure diagnostic color drift.
    """
    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    proc_lab = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    
    delta_e = np.sqrt(np.sum((orig_lab - proc_lab) ** 2, axis=2))
    return float(np.mean(delta_e))

def compute_noise_level(image_np: np.ndarray) -> float:
    """
    Estimates standard deviation of high-frequency noise bands 
    using Laplacian filtering.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(lap))

def compute_masked_metrics(original_bgr: np.ndarray, processed_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calculates sharpness and SSIM strictly inside the predicted lesion boundary mask.
    """
    if mask is None or np.sum(mask) == 0:
        return {
            "lesion_sharpness": compute_laplacian_variance(processed_bgr),
            "lesion_ssim": compute_ssim(original_bgr, processed_bgr)
        }
    
    binary_mask = (mask > 128).astype(np.uint8)
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return {
            "lesion_sharpness": compute_laplacian_variance(processed_bgr),
            "lesion_ssim": compute_ssim(original_bgr, processed_bgr)
        }
        
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    
    crop_orig = original_bgr[ymin:ymax+1, xmin:xmax+1]
    crop_proc = processed_bgr[ymin:ymax+1, xmin:xmax+1]
    
    return {
        "lesion_sharpness": compute_laplacian_variance(crop_proc),
        "lesion_ssim": compute_ssim(crop_orig, crop_proc)
    }

def compute_all_metrics(
    original_bgr: np.ndarray, 
    processed_bgr: np.ndarray, 
    mask: np.ndarray = None,
    orig_confidence: float = 0.0,
    proc_confidence: float = 0.0
) -> dict:
    """
    Master calculator function returning all DermaLens reliability metrics safely.
    """
    if original_bgr.shape != processed_bgr.shape:
        processed_bgr = cv2.resize(processed_bgr, (original_bgr.shape[1], original_bgr.shape[0]))

    lap_var = sanitize_float(compute_laplacian_variance(processed_bgr))
    psnr_score = compute_psnr(original_bgr, processed_bgr)
    ssim_score = compute_ssim(original_bgr, processed_bgr)
    rmse_score = compute_rmse(original_bgr, processed_bgr)
    
    color_shift = sanitize_float(compute_color_shift_delta_e(original_bgr, processed_bgr))
    noise_level = sanitize_float(compute_noise_level(processed_bgr))
    
    masked_res = compute_masked_metrics(original_bgr, processed_bgr, mask)
    
    total_pixels = original_bgr.shape[0] * original_bgr.shape[1]
    mask_pixels = np.sum(mask > 128) if mask is not None else 0
    coverage_ratio = float(mask_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    confidence_delta = sanitize_float(proc_confidence - orig_confidence)

    return {
        "laplacian_variance": round(lap_var, 2),
        "psnr": round(psnr_score, 2),
        "ssim": round(ssim_score, 4),
        "rmse": round(rmse_score, 2),
        "color_shift_delta_e": round(color_shift, 2),
        "noise_level": round(noise_level, 2),
        "lesion_sharpness": round(sanitize_float(masked_res["lesion_sharpness"]), 2),
        "lesion_ssim": round(sanitize_float(masked_res["lesion_ssim"], default=1.0), 4),
        "lesion_coverage_ratio": round(sanitize_float(coverage_ratio), 4),
        "classifier_confidence_delta": round(confidence_delta, 4)
    }