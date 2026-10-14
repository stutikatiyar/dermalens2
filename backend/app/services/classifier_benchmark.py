import numpy as np


def benchmark_classifier(image, predictor_before, predictor_after):
    """
    Both predictors must return class probabilities or scores
    in the same class order.

    This measures prediction changes, not accuracy by itself.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    before = np.asarray(predictor_before(image), dtype=float)
    after = np.asarray(predictor_after(image), dtype=float)

    if before.ndim != 1 or after.ndim != 1:
        raise ValueError("Predictors must return one-dimensional scores.")

    if before.shape != after.shape:
        raise ValueError("Predictors must return matching class scores.")

    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Predictions contain invalid numeric values.")

    return {
        "predicted_class_before": int(np.argmax(before)),
        "predicted_class_after": int(np.argmax(after)),
        "prediction_changed": bool(
            np.argmax(before) != np.argmax(after)
        ),
        "scores_before": before.tolist(),
        "scores_after": after.tolist(),
    }