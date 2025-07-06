from fastapi import  HTTPException, FastAPI, UploadFile, File
from optimum.pipelines import pipeline
import logging
from PIL import Image
import io
import time
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

classifier: Optional[pipeline] = None

@app.on_event("startup")
async def load_model():
    """Load the classification model at startup"""
    global classifier
    try:
        classifier = pipeline(
            "image-classification",
            model="onnx-community/nsfw_image_detection-ONNX",
            device=-1,
            accelerator="ort",
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}


@app.post("/api/image_classify")
async def image_classify(
    file: UploadFile = File(),
):
    """Classify image"""
    try:
        start_time = time.time()
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_rgb = img.convert("RGB")
        predictions = classifier(img_rgb)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "type": "single_image",
            "predictions": predictions,
            #"detected": predictions[0]["score"] if predictions else 0.0,
            "processing_time": processing_time,
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")