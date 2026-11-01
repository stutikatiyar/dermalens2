
"""
MPRNet deblurring adapter for DermaLens.

Expected layout:
project_root/
  third_party/MPRNet/Deblurring/MPRNet.py
  third_party/MPRNet/Deblurring/checkpoints/deblurring/model_deblurring.pth
  backend/app/ml/mprnet_adapter.py
"""

from functools import lru_cache
from pathlib import Path
import sys
import time

import cv2
import numpy as np


# Support both local execution and the Docker layout.
_THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT_CANDIDATES = [
    _THIS_FILE.parents[2],  # Docker: /app
    _THIS_FILE.parents[3],  # Local: project root
]

PROJECT_ROOT = next(
    (
        path for path in PROJECT_ROOT_CANDIDATES
        if (
            path / "third_party" / "MPRNet"
            / "Deblurring" / "MPRNet.py"
        ).is_file()
    ),
    PROJECT_ROOT_CANDIDATES[0],
)

MPRNET_DIR = (
    PROJECT_ROOT / "third_party" / "MPRNet" / "Deblurring"
)

WEIGHTS_PATH = (
    MPRNET_DIR
    / "checkpoints"
    / "deblurring"
    / "model_deblurring.pth"
)

DEVICE_NAME = "cpu"


@lru_cache(maxsize=1)
def _load_model():
    """Load the official MPRNet architecture and checkpoint."""

    if not (MPRNET_DIR / "MPRNet.py").is_file():
        raise FileNotFoundError(
            f"MPRNet source not found: {MPRNET_DIR / 'MPRNet.py'}"
        )

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"MPRNet checkpoint not found: {WEIGHTS_PATH}"
        )

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

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "params" in checkpoint:
            state = checkpoint["params"]
        elif (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        )

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint did not contain a state dictionary."
        )

    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "MPRNet checkpoint does not match the Deblurring architecture."
        ) from exc

    model.to(DEVICE_NAME)
    model.eval()

    print(f"MPRNET -> loaded checkpoint: {WEIGHTS_PATH}")
    print(f"MPRNET -> source directory: {MPRNET_DIR}")
    print(f"MPRNET -> device: {DEVICE_NAME}")

    return model


def enhance_mprnet_bgr(image_bgr):
    """
    Deblur one OpenCV BGR uint8 image.

    Returns:
        restored_bgr_uint8, metadata
    """

    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError(
            "enhance_mprnet_bgr expects a valid OpenCV image."
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    import torch
    import torch.nn.functional as F

    image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    height, width = image_bgr.shape[:2]
    started = time.perf_counter()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .float()
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    factor = 8

    padded_h = ((height + factor - 1) // factor) * factor
    padded_w = ((width + factor - 1) // factor) * factor

    pad_h = padded_h - height
    pad_w = padded_w - width

    if pad_h or pad_w:
        mode = (
            "reflect"
            if pad_h < height and pad_w < width
            else "replicate"
        )
        tensor = F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode=mode,
        )

    model = _load_model()

    with torch.inference_mode():
        outputs = model(tensor)

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

    rgb_out = np.clip(
        np.round(rgb_out * 255.0),
        0,
        255,
    ).astype(np.uint8)

    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    metadata = {
        "method": "MPRNet Deblurring",
        "device": DEVICE_NAME,
        "input_size": f"{width}x{height}",
        "output_size": f"{bgr_out.shape[1]}x{bgr_out.shape[0]}",
        "elapsed_ms": round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
    }

    print("MPRNET ->", metadata)

    return bgr_out, metadata
