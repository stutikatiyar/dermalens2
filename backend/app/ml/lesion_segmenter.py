"""

DermaLens — Lesion Segmentation Module



Pretrained SegFormer MiT-B0 model fine-tuned for

ISIC 2018 skin-lesion segmentation.



Pipeline:



    Dermatology Image

          ↓

    SegFormer MiT-B0

          ↓

    Lesion Probability Map

          ↓

    Binary Lesion Mask

          ↓

    Mask Post-processing

          ↓

    Lesion Region Information

"""



from pathlib import Path



import cv2

import numpy as np

import torch



from transformers import (

    SegformerConfig,

    SegformerForSemanticSegmentation,

)

from huggingface_hub import hf_hub_download





# ============================================================

# CONFIGURATION

# ============================================================



MODEL_REPOSITORY = "mahnoor-2722/isic-segformer"



MODEL_FILENAME = "best_segformer_isic2018.pth"



BASE_MODEL = "nvidia/mit-b0"



INPUT_SIZE = 256



DEFAULT_MASK_THRESHOLD = 0.50



DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)





# ============================================================

# MODEL CACHE

# ============================================================



_model = None





# ============================================================

# MODEL LOADING

# ============================================================



def load_segmentation_model():

    """

    Load the pretrained ISIC SegFormer model.



    The checkpoint is loaded directly as a PyTorch state_dict.



    strict=True is intentionally used so that a mismatch between

    the architecture and checkpoint cannot silently produce an

    invalid segmentation model.

    """



    global _model



    if _model is not None:

        return _model



    print(

        f"[LesionSegmentation] Loading model on {DEVICE}..."

    )



    # --------------------------------------------------------

    # Get the pretrained checkpoint

    # --------------------------------------------------------



    weights_path = hf_hub_download(

        repo_id=MODEL_REPOSITORY,

        filename=MODEL_FILENAME,

    )



    print(

        "[LesionSegmentation] Loading checkpoint from:"

    )

    print(weights_path)



    # --------------------------------------------------------

    # Create an explicit binary SegFormer configuration

    # --------------------------------------------------------



    base_config = SegformerConfig.from_pretrained(

        BASE_MODEL

    )



    # Binary segmentation:

    #

    # 0 = background

    # 1 = lesion

    #

    # IMPORTANT:

    # Do not inherit the ImageNet 1000-class mapping.

    base_config.num_labels = 1



    base_config.id2label = {

        0: "lesion"

    }



    base_config.label2id = {

        "lesion": 0

    }



    # --------------------------------------------------------

    # Create segmentation model

    # --------------------------------------------------------



    model = SegformerForSemanticSegmentation(

        base_config

    )



    # --------------------------------------------------------

    # Load checkpoint

    # --------------------------------------------------------



    checkpoint = torch.load(

        weights_path,

        map_location="cpu",

    )



    if not isinstance(checkpoint, dict):

        raise RuntimeError(

            "Segmentation checkpoint is not a state dictionary."

        )



    print(

        f"[LesionSegmentation] "

        f"Checkpoint contains {len(checkpoint)} parameters."

    )



    # --------------------------------------------------------

    # Direct state-dict loading

    # --------------------------------------------------------



    load_result = model.load_state_dict(

        checkpoint,

        strict=False,

    )



    missing_keys = list(

        load_result.missing_keys

    )



    unexpected_keys = list(

        load_result.unexpected_keys

    )



    print(

        f"[LesionSegmentation] "

        f"Missing keys: {len(missing_keys)}"

    )



    print(

        f"[LesionSegmentation] "

        f"Unexpected keys: {len(unexpected_keys)}"

    )



    # --------------------------------------------------------

    # NEVER accept a partially loaded segmentation model

    # --------------------------------------------------------



    if missing_keys or unexpected_keys:



        print(

            "\n[LesionSegmentation] MODEL LOAD FAILED"

        )



        if missing_keys:

            print("\nMissing keys:")

            for key in missing_keys:

                print(f"  {key}")



        if unexpected_keys:

            print("\nUnexpected keys:")

            for key in unexpected_keys:

                print(f"  {key}")



        raise RuntimeError(

            "Segmentation checkpoint does not exactly match "

            "the SegFormer architecture."

        )



    # --------------------------------------------------------

    # Move model to CPU/GPU

    # --------------------------------------------------------



    model.to(DEVICE)



    model.eval()



    _model = model



    print(

        "[LesionSegmentation] "

        "SegFormer model loaded successfully."

    )



    print(

        "[LesionSegmentation] "

        "All checkpoint parameters matched."

    )



    return _model





# ============================================================

# INPUT VALIDATION

# ============================================================



def validate_image(image):

    """

    Validate an OpenCV BGR image.

    """



    if image is None:

        raise ValueError(

            "Lesion segmentation received an empty image."

        )



    if not isinstance(image, np.ndarray):

        raise ValueError(

            "Lesion segmentation expects a NumPy image."

        )



    if image.size == 0:

        raise ValueError(

            "Lesion segmentation received an empty image array."

        )



    if len(image.shape) != 3:

        raise ValueError(

            "Lesion segmentation expects a 3-channel image."

        )



    if image.shape[2] != 3:

        raise ValueError(

            "Lesion segmentation expects a BGR image with 3 channels."

        )



    return True





# ============================================================

# PREPROCESSING

# ============================================================



def preprocess_image(image):

    """

    Convert an OpenCV BGR image into the normalized

    tensor expected by SegFormer.

    """



    validate_image(image)



    # BGR → RGB

    rgb = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB,

    )



    # Resize to SegFormer input resolution.

    resized = cv2.resize(

        rgb,

        (INPUT_SIZE, INPUT_SIZE),

        interpolation=cv2.INTER_LINEAR,

    )



    # Convert to [0, 1].

    tensor = (

        resized.astype(np.float32)

        / 255.0

    )



    # ImageNet normalization.

    mean = np.array(

        [0.485, 0.456, 0.406],

        dtype=np.float32,

    )



    std = np.array(

        [0.229, 0.224, 0.225],

        dtype=np.float32,

    )



    tensor = (

        tensor - mean

    ) / std



    # HWC → CHW

    tensor = np.transpose(

        tensor,

        (2, 0, 1),

    )



    tensor = torch.from_numpy(

        tensor

    ).float()



    # Add batch dimension.

    tensor = tensor.unsqueeze(0)



    return tensor.to(DEVICE)





# ============================================================

# PROBABILITY MAP

# ============================================================



def generate_lesion_probability_map(

    image,

    model=None,

):

    """

    Generate a lesion probability map.



    Returns:

        2D NumPy array with values between 0 and 1.

    """



    validate_image(image)



    if model is None:

        model = load_segmentation_model()



    original_height, original_width = image.shape[:2]



    inputs = preprocess_image(image)



    with torch.no_grad():



        outputs = model(

            pixel_values=inputs

        )



        logits = outputs.logits



        # Binary segmentation:

        # [batch, 1, height, width]

        logits = logits[:, 0, :, :]



        probabilities = torch.sigmoid(

            logits

        )



    probability_map = (

        probabilities[0]

        .cpu()

        .numpy()

    )



    # Resize prediction back to original image size.

    probability_map = cv2.resize(

        probability_map,

        (

            original_width,

            original_height,

        ),

        interpolation=cv2.INTER_LINEAR,

    )



    probability_map = np.clip(

        probability_map,

        0.0,

        1.0,

    )



    return probability_map





# ============================================================

# MASK POST-PROCESSING

# ============================================================



def postprocess_mask(

    probability_map,

    threshold=DEFAULT_MASK_THRESHOLD,

):

    """

    Convert lesion probabilities into a binary mask.



    0   → background

    255 → lesion

    """



    if probability_map is None:

        raise ValueError(

            "Probability map cannot be None."

        )



    if probability_map.ndim != 2:

        raise ValueError(

            "Probability map must be a 2D array."

        )



    mask = (

        probability_map >= threshold

    ).astype(

        np.uint8

    ) * 255



    # Small morphological cleanup.

    kernel = np.ones(

        (3, 3),

        np.uint8,

    )



    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

    )



    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

    )



    return mask





# ============================================================

# LESION MASK

# ============================================================



def generate_lesion_mask(

    image,

    model=None,

    threshold=DEFAULT_MASK_THRESHOLD,

):

    """

    Generate a binary lesion mask.

    """



    probability_map = (

        generate_lesion_probability_map(

            image,

            model=model,

        )

    )



    mask = postprocess_mask(

        probability_map,

        threshold=threshold,

    )



    return mask





# ============================================================

# LESION REGION INFORMATION

# ============================================================



def get_lesion_region(mask):

    """

    Extract lesion area, area ratio and bounding box.

    """



    if mask is None:

        raise ValueError(

            "Lesion mask cannot be None."

        )



    if mask.ndim != 2:

        raise ValueError(

            "Lesion mask must be a 2D array."

        )



    height, width = mask.shape



    lesion_pixels = np.count_nonzero(

        mask

    )



    if lesion_pixels == 0:



        return {

            "detected": False,

            "area_pixels": 0,

            "area_ratio": 0.0,

            "bbox": None,

        }



    area_ratio = (

        lesion_pixels

        / float(height * width)

    )



    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE,

    )



    if not contours:



        return {

            "detected": False,

            "area_pixels": int(

                lesion_pixels

            ),

            "area_ratio": float(

                area_ratio

            ),

            "bbox": None,

        }



    largest_contour = max(

        contours,

        key=cv2.contourArea,

    )



    x, y, w, h = cv2.boundingRect(

        largest_contour

    )



    return {

        "detected": True,

        "area_pixels": int(

            lesion_pixels

        ),

        "area_ratio": float(

            area_ratio

        ),

        "bbox": {

            "x": int(x),

            "y": int(y),

            "width": int(w),

            "height": int(h),

        },

    }





# ============================================================

# MASK OVERLAY

# ============================================================



def create_mask_overlay(image, mask):
    """Create an overlay showing the predicted lesion region."""
    validate_image(image)

    if mask is None:
        raise ValueError("Mask cannot be None.")

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D NumPy array.")

    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = image.copy()
    lesion_region = mask > 0

    if not np.any(lesion_region):
        return overlay

    # Green is used only for visualization/debugging.
    highlight = np.zeros_like(image)
    highlight[:] = (0, 255, 0)  # OpenCV uses BGR ordering.

    # Blend full 3-channel images first, then apply to masked pixels.
    blended = cv2.addWeighted(image, 0.65, highlight, 0.35, 0)
    overlay[lesion_region] = blended[lesion_region]

    return overlay



# ============================================================

# COMPLETE SEGMENTATION PIPELINE

# ============================================================



def segment_lesion(

    image,

    threshold=DEFAULT_MASK_THRESHOLD,

):

    """

    Complete lesion segmentation pipeline.



    Returns:

        probability_map

        binary mask

        lesion information

        visualization overlay

    """



    validate_image(image)



    model = load_segmentation_model()



    probability_map = (

        generate_lesion_probability_map(

            image,

            model=model,

        )

    )



    mask = postprocess_mask(

        probability_map,

        threshold=threshold,

    )



    lesion_info = get_lesion_region(

        mask

    )



    overlay = create_mask_overlay(

        image,

        mask,

    )



    return {

        "probability_map": probability_map,

        "mask": mask,

        "lesion_info": lesion_info,

        "overlay": overlay,

    }
