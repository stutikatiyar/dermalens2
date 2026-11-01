"""Isolated MPRNet lesion-crop experiment for DermaLens.
Run from the backend folder with the image path as the first argument.
This does not modify the upload route or production selection.
"""
from pathlib import Path
import sys
import cv2

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parents[1]


sys.path.insert(0, str(BACKEND_DIR))

from app.ml.lesion_segmenter import segment_lesion
from app.ml.mprnet_adapter import enhance_mprnet_bgr
from app.ml.lesion_crop_experiment import enhance_lesion_crop, save_lesion_crop_debug


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Pass the input image path as the first argument.")

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")

    original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original is None:
        raise SystemExit(f"OpenCV could not read: {input_path}")

    print("INPUT:", input_path)
    print("IMAGE SIZE:", f"{original.shape[1]}x{original.shape[0]}")

    segmentation = segment_lesion(original)
    mask = segmentation.get("mask")
    lesion_info = segmentation.get("lesion_info", {})
    print("SEGMENTATION INFO:", lesion_info)

    if mask is None:
        raise SystemExit("Segmentation returned no mask; crop experiment stopped.")

    output = enhance_lesion_crop(
        original_bgr=original,
        lesion_mask=mask,
        restore_fn=enhance_mprnet_bgr,
        margin_ratio=1.5,
        min_crop_size=128,
        blend_strength=0.85,
    )

    output_dir = BACKEND_DIR / "debug" / "lesion_crop"
    paths = save_lesion_crop_debug(output, output_dir)

    original_path = output_dir / "original_full.png"
    if not cv2.imwrite(str(original_path), original):
        raise SystemExit(f"Could not save original image: {original_path}")

    print("\nLESION CROP TEST PASSED")
    print("LESION BBOX:", output["lesion_bbox"])
    print("CROP BBOX:", output["bbox"])
    print("RESTORATION METADATA:", output["metadata"])
    print("DEBUG OUTPUTS:")
    print("  Original full image:", original_path)
    for key, value in paths.items():
        print(f"  {key}: {value}")
    print("\nInspect restored_crop_raw.png separately from blended_full.png.")
    print("A sharper-looking result is not proof of true medical detail recovery.")


if __name__ == "__main__":
    main()
