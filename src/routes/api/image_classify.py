from fastapi import  HTTPException, APIRouter, UploadFile, File
from nsfw_image_detector import NSFWDetector
from PIL import Image
import io
import time
from src.shared.resize_image import resize_image

router = APIRouter()

@router.post("/api/image_classify")
async def image_classify(
    file: UploadFile = File(),
):
    """Classify test image"""
    try:
        detector = NSFWDetector()
        start_time = time.time()
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_rgb = img.convert("RGB")
        img_resized = resize_image(img_rgb, size=(224, 224))
        is_nsfw = detector.is_nsfw(img_resized)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "type": "single_image",
            "isNSFW": is_nsfw,
            "processing_time": processing_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing image")