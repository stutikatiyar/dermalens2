"""Safe helpers for basic file operations."""
from pathlib import Path


def ensure_directory(path):
    """Create a directory if needed and return its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(filename):
    """Return a lowercase extension, including the leading dot."""
    return Path(filename).suffix.lower()


def is_allowed_image(filename):
    """Check whether a filename has a supported image extension."""
    return get_file_extension(filename) in {
        ".jpg", ".jpeg", ".png", ".webp"
    }


def safe_filename(filename):
    """Strip directory components from a user-provided filename."""
    return Path(filename.replace("\\", "/")).name


def save_bytes(data, destination):
    """Save bytes to a destination, creating parent directories."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
