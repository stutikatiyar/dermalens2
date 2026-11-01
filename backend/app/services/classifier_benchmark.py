import numpy as np


def benchmark_classifier(image_before, image_after, predictor, ground_truth=None):
    """
    Evaluates downstream classifier behaviour across degraded and enhanced images.

    Args:
        image_before (numpy.ndarray): Degraded input image array.
        image_after (numpy.ndarray): Enhanced output image array.
        predictor (callable): Model inference function taking an image array and 
                             returning a 1D numpy array of probabilities or scores.
        ground_truth (int, optional): True class index label. If provided, calculates
                                     accuracy improvements.

    Returns:
        dict: Complete comparison metrics including class predictions, confidence scores,
              prediction shift indicators, and accuracy flags (if ground_truth given).
    """
    if image_before is None or image_before.size == 0:
        raise ValueError("Input image_before is empty.")
    if image_after is None or image_after.size == 0:
        raise ValueError("Input image_after is empty.")

    # Obtain class score vectors for both states
    before = np.asarray(predictor(image_before), dtype=float)
    after = np.asarray(predictor(image_after), dtype=float)

    if before.ndim != 1 or after.ndim != 1:
        raise ValueError("Predictors must return one-dimensional scores.")

    if before.shape != after.shape:
        raise ValueError("Predictors must return matching class scores.")

    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Predictions contain invalid numeric values.")

    # Class indices with highest confidence
    class_before = int(np.argmax(before))
    class_after = int(np.argmax(after))

    # Top-1 confidence values
    conf_before = float(before[class_before])
    conf_after = float(after[class_after])

    results = {
        "predicted_class_before": class_before,
        "predicted_class_after": class_after,
        "prediction_changed": bool(class_before != class_after),
        "confidence_before": conf_before,
        "confidence_after": conf_after,
        "confidence_delta": float(conf_after - conf_before),
        "scores_before": before.tolist(),
        "scores_after": after.tolist(),
    }

    # Evaluate accuracy changes if ground truth label is supplied
    if ground_truth is not None:
        gt = int(ground_truth)
        correct_before = bool(class_before == gt)
        correct_after = bool(class_after == gt)

        results["ground_truth"] = gt
        results["correct_before"] = correct_before
        results["correct_after"] = correct_after
        results["accuracy_improved"] = bool(not correct_before and correct_after)
        results["accuracy_degraded"] = bool(correct_before and not correct_after)

    return results