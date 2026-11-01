"""Run an isolated MPRNet smoke test without changing DermaLens routes."""
from pathlib import Path
import sys
import cv2

# Run from the project root:
# backend\venv\Scripts\python.exe backend\app\ml\test_mprnet.py path\to\sample.jpg
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mprnet_adapter import enhance_mprnet_bgr

if len(sys.argv) < 2:
    raise SystemExit(
        'Usage: python backend/app/ml/test_mprnet.py "path\\to\\sample.jpg"'
    )

source_path = Path(sys.argv[1]).resolve()
if not source_path.is_file():
    raise SystemExit(f"Input image not found: {source_path}")

image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
if image is None:
    raise SystemExit(f"OpenCV could not read: {source_path}")

try:
    restored, metadata = enhance_mprnet_bgr(image)
except Exception as exc:
    print(f"MPRNET TEST FAILED: {type(exc).__name__}: {exc}")
    raise

output_path = source_path.with_name(source_path.stem + "_mprnet.png")
if not cv2.imwrite(str(output_path), restored):
    raise SystemExit(f"Could not save output image: {output_path}")

print("MPRNET TEST PASSED")
print("Saved output:", output_path)
print("Metadata:", metadata)
