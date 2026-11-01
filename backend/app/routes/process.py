from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.image_service import ImageService
from app.ml.pipeline import process_image
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/process")
async def process_uploaded_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = ImageService.decode_bytes(data)
        result = process_image(image)
        result["filename"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image processing failed")
        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Check backend logs."
        ) from exc
    finally:
        await file.close()
