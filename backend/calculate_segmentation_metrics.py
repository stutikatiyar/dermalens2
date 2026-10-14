import cv2
import numpy as np
import os

GROUND_TRUTH_DIR = "segmentation_ground_truth"
PREDICTED_DIR = "segmentation_test"

IMAGE_IDS = [
    "ISIC_0024306",
    "ISIC_0024307",
    "ISIC_0024308",
    "ISIC_0024309",
    "ISIC_0024310",
]


def calculate_dice(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(pred, gt).sum()

    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator


def calculate_iou(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


results = []

for image_id in IMAGE_IDS:

    gt_path = os.path.join(
        GROUND_TRUTH_DIR,
        f"{image_id}_UNET.png"
    )

    pred_path = os.path.join(
        PREDICTED_DIR,
        f"{image_id}_lesion_mask.png"
    )

    if not os.path.exists(gt_path):
        print(f"❌ Ground truth missing: {gt_path}")
        continue

    if not os.path.exists(pred_path):
        print(f"❌ Prediction missing: {pred_path}")
        continue

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        print(f"❌ Could not read ground truth: {gt_path}")
        continue

    if pred is None:
        print(f"❌ Could not read prediction: {pred_path}")
        continue

    print(f"\n{image_id}")
    print(f"  Ground truth shape: {gt.shape}")
    print(f"  Prediction shape:   {pred.shape}")

    if gt.shape != pred.shape:
        print("  ❌ Shape mismatch — skipping")
        continue

    dice = calculate_dice(pred, gt)
    iou = calculate_iou(pred, gt)

    results.append((image_id, dice, iou))

    print(f"  Dice: {dice:.4f}")
    print(f"  IoU : {iou:.4f}")


if results:

    mean_dice = np.mean([r[1] for r in results])
    mean_iou = np.mean([r[2] for r in results])

    print("\n" + "=" * 50)
    print("SEGFORMER SEGMENTATION EVALUATION")
    print("=" * 50)

    for image_id, dice, iou in results:
        print(
            f"{image_id}: "
            f"Dice = {dice:.4f}, "
            f"IoU = {iou:.4f}"
        )

    print("-" * 50)
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")
    print("=" * 50)

else:
    print("\n❌ No images could be evaluated.")