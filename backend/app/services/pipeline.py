from app.services.image_loader import load_image
from app.services.quality_analyzer import analyze_quality
from app.services.enhancement_engine import enhance_image
from app.services.metrics_generator import generate_metrics


def process_image(
    source,
    segmenter=None,
    classifier_before=None,
    classifier_after=None,
):
    """
    Run the DermaLens preprocessing pipeline.

    Returns the original image, enhanced image, quality report,
    image-comparison metrics, and optional model results.
    """
    original = load_image(source)

    quality_before = analyze_quality(original)

    enhanced = enhance_image(
        original,
        quality=quality_before,
    )

    quality_after = analyze_quality(enhanced)

    metrics = generate_metrics(original, enhanced)

    result = {
        "quality_before": quality_before,
        "quality_after": quality_after,
        "metrics": metrics,
        "enhancement_applied": not (
            original == enhanced
        ).all(),
        "enhanced_image": enhanced,
        "segmentation_mask": None,
        "classifier_benchmark": None,
    }

    if segmenter is not None:
        result["segmentation_mask"] = segmenter.segment(enhanced)

    if classifier_before is not None and classifier_after is not None:
        from app.services.classifier_benchmark import (
            benchmark_classifier,
        )

        result["classifier_benchmark"] = benchmark_classifier(
            original,
            classifier_before,
            classifier_after,
        )

    return result