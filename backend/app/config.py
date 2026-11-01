import os

APP_NAME = os.getenv("APP_NAME", "DermaLens")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
