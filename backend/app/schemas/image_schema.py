from typing import Any, Optional
from pydantic import BaseModel, Field

class ImageMetrics(BaseModel):
    original_sharpness: Optional[float] = None
    enhanced_sharpness: Optional[float] = None
    original_contrast: Optional[float] = None
    enhanced_contrast: Optional[float] = None
    original_brightness: Optional[float] = None
    enhanced_brightness: Optional[float] = None
    brightness_improvement: Optional[float] = None
    quality_score: Optional[float] = None

class ImageProcessResponse(BaseModel):
    success: bool = True
    message: str = "Image processed"
    image: Optional[str] = None
    mode: str = "enhancement"
    metrics: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

class ImageUploadInfo(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size_bytes: int
