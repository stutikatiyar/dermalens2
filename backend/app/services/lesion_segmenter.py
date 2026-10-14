import numpy as np


class LesionSegmenter:
    """
    Adapter for the existing DermaLens segmentation model.

    Pass a predictor callable that accepts an RGB NumPy image
    and returns a segmentation mask.
    """

    def __init__(self, predictor):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")

        self.predictor = predictor

    def segment(self, image):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        mask = self.predictor(image)

        if mask is None:
            raise RuntimeError("Segmentation model returned no mask.")

        mask = np.asarray(mask)

        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError("Expected a two-dimensional segmentation mask.")

        return mask