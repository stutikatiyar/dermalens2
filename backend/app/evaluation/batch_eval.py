"""Batch evaluation helpers for DermaLens image processing."""
import csv
from pathlib import Path

import cv2

from app.ml.pipeline import process_image


def evaluate_directory(input_dir, output_csv="evaluation_results.csv"):
    """Process images in a directory and record basic evaluation metrics."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    results = []
    for path in image_paths:
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Image could not be decoded")

            result = process_image(image)
            metrics = result.get("metrics", {})
            results.append({
                "filename": str(path),
                "status": "success",
                **metrics,
                "error": "",
            })
        except Exception as exc:
            results.append({
                "filename": str(path),
                "status": "failed",
                "error": str(exc),
            })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = sorted({key for row in results for key in row})
    if not columns:
        columns = ["filename", "status", "error"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    return {
        "total": len(image_paths),
        "successful": sum(r["status"] == "success" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "output_csv": str(output_path.resolve()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DermaLens on a folder")
    parser.add_argument("input_dir", help="Directory containing test images")
    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="CSV report path",
    )
    args = parser.parse_args()
    print(evaluate_directory(args.input_dir, args.output))
