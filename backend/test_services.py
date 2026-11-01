
import numpy as np

from app.services.quality_analyzer import analyze_quality
from app.services.enhancement_engine import enhance_image
from app.services.metrics_generator import generate_metrics


def test_image_services():
    """
    Smoke test for the DermaLens image-processing services.
    Uses a synthetic image, not a clinical test image.
    """

    # Create a synthetic RGB image
    image = np.full((256, 256, 3), 120, dtype=np.uint8)
    image[80:176, 80:176] = 160

    # 1. Analyze original image quality
    quality = analyze_quality(image)

    assert isinstance(quality, dict)
    assert "brightness" in quality
    assert "sharpness" in quality
    assert "issues" in quality

    # 2. Enhance the image
    enhanced = enhance_image(image, quality=quality)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8

    # 3. Generate comparison metrics
    metrics = generate_metrics(image, enhanced)

    assert isinstance(metrics, dict)
    assert "psnr_db" in metrics
    assert "ssim" in metrics
    assert 0 <= metrics["ssim"] <= 1

    # Display results
    print("\n--- DermaLens Service Test ---")
    print("Original image shape:", image.shape)
    print("Enhanced image shape:", enhanced.shape)
    print("Quality analysis:", quality)
    print("Comparison metrics:", metrics)
    print("\nAll basic service checks passed!")


if __name__ == "__main__":
    test_image_services()