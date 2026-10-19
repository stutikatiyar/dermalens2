"""
MPRNet deblurring adapter for DermaLens.

Expected layout:
project_root/
  third_party/MPRNet/Deblurring/MPRNet.py
  backend/app/ml/mprnet_adapter.py
  backend/app/ml/mprnet/weights/model_deblurring.pth

This module is experimental and is not connected to the production pipeline yet.
"""
from functools import lru_cache
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MPRNET_DIR = PROJECT_ROOT / "third_party" / "MPRNet" / "Deblurring"
WEIGHTS_PATH = (
    Path(__file__).resolve().parent
    / "mprnet"
    / "weights"
    / "model_deblurring.pth"
)
DEVICE_NAME = "cpu"


@lru_cache(maxsize=1)
def _load_model():
    """Load the official MPRNet architecture and deblurring checkpoint."""
    if not (MPRNET_DIR / "MPRNet.py").is_file():
        raise FileNotFoundError(
            "MPRNet source was not found. Expected: "
            f"{MPRNET_DIR / 'MPRNet.py'}. "
            "Clone swz30/MPRNet into third_party/MPRNet."
        )
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"MPRNet checkpoint not found: {WEIGHTS_PATH}"
        )

    # Official MPRNet files use local imports (e.g. subnet, utils).
    mprnet_dir_text = str(MPRNET_DIR)
    if mprnet_dir_text not in sys.path:
        sys.path.insert(0, mprnet_dir_text)

    import torch
    from MPRNet import MPRNet

    model = MPRNet()
    checkpoint = torch.load(
        str(WEIGHTS_PATH),
        map_location="cpu",
        weights_only=False,
    )

    # Support common checkpoint formats without silently accepting
    # incompatible parameter names.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "params" in checkpoint:
            state = checkpoint["params"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        )

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint did not contain a state dictionary.")

    # Remove DataParallel's prefix if present.
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "MPRNet checkpoint keys do not match the official Deblurring "
            "architecture. Check that model_deblurring.pth is the official "
            "MPRNet deblurring checkpoint and that the repository is intact."
        ) from exc

    model.to(DEVICE_NAME)
    model.eval()
    print(f"MPRNET → loaded checkpoint: {WEIGHTS_PATH}")
    print("MPRNET → device: CPU")
    return model


def enhance_mprnet_bgr(image_bgr):
    """
    Deblur one OpenCV BGR uint8 image.
    Returns (restored_bgr_uint8, metadata).
    MPRNet may alter image content; compare output before using clinically.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("enhance_mprnet_bgr expects a valid OpenCV image.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    import torch
    import torch.nn.functional as F

    image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    height, width = image_bgr.shape[:2]
    started = time.perf_counter()

    # MPRNet's official image-deblurring code operates on RGB floats [0, 1].
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .float()
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    # Pad bottom/right to multiples of 8, then crop back after inference.
    factor = 8
    padded_h = ((height + factor - 1) // factor) * factor
    padded_w = ((width + factor - 1) // factor) * factor
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h or pad_w:
        # Reflect padding is suitable only when each padding amount is
        # smaller than the corresponding dimension; use replicate otherwise.
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)

    model = _load_model()
    with torch.inference_mode():
        outputs = model(tensor)

    # Official MPRNet test code uses restored[0].
    if isinstance(outputs, (list, tuple)):
        restored = outputs[0]
    else:
        restored = outputs

    restored = torch.clamp(restored, 0.0, 1.0)
    restored = restored[:, :, :height, :width]
    rgb_out = (
        restored[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    rgb_out = np.clip(np.round(rgb_out * 255.0), 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    metadata = {
        "method": "MPRNet Deblurring",
        "device": DEVICE_NAME,
        "input_size": f"{width}x{height}",
        "output_size": f"{bgr_out.shape[1]}x{bgr_out.shape[0]}",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    print("MPRNET →", metadata)
    return bgr_out, metadata
