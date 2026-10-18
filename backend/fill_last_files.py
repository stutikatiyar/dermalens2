from pathlib import Path

files = {
    "app/evaluation/batch_eval.py": r'''
"""Batch evaluation of DermaLens image enhancement."""

import csv
from pathlib import Path

import cv2

from app.ml.pipeline import process_image


def evaluate_directory(input_dir, output_csv="evaluation_results.csv"):
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise ValueError(f"Directory not found: {input_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    results = []

    for path in images:
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Unable to read image")

            result = process_image(image)

            row = {
                "filename": str(path),
                "status": "success",
                "error": "",
            }
            row.update(result.get("metrics", {}))
            results.append(row)

        except Exception as exc:
            results.append({
                "filename": str(path),
                "status": "failed",
                "error": str(exc),
            })

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = sorted({key for row in results for key in row})
    if not columns:
        columns = ["filename", "status", "error"]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    return {
        "total": len(images),
        "successful": sum(r["status"] == "success" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "report": str(output.resolve()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate DermaLens on a folder of images"
    )
    parser.add_argument("input_dir", help="Folder containing test images")
    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Output CSV report",
    )

    args = parser.parse_args()
    print(evaluate_directory(args.input_dir, args.output))
''',

    "app/services/__init__.py": r'''
"""Service layer for DermaLens."""

from app.services.image_service import ImageService

__all__ = ["ImageService"]
''',

    "app/utils/file_utils.py": r'''
"""File utilities used by the DermaLens backend."""

from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_directory(path):
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(filename):
    """Return the lowercase file extension."""
    return Path(filename).suffix.lower()


def is_allowed_image(filename):
    """Check whether a filename has a supported image extension."""
    return get_file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS


def safe_filename(filename):
    """Remove directory components from a supplied filename."""
    return Path(str(filename).replace("\\", "/")).name


def save_bytes(data, destination):
    """Save bytes to a file and create its parent directory."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
'''
}

for filename, content in files.items():
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"SKIPPED (already contains code): {path}")
        continue

    path.write_text(content.lstrip(), encoding="utf-8")
    print(f"CREATED: {path}")

print("\nFinished writing the files.")
