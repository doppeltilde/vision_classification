from dotenv import load_dotenv
import os, urllib.request
from optimum.pipelines import pipeline
import logging

logger = logging.getLogger(__name__)

load_dotenv()

global classifier
classifier = None

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv(
    "DEFAULT_MODEL_NAME", "onnx-community/nsfw_image_detection-ONNX"
)

# https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
default_tflite_model_url = os.getenv(
    "DEFAULT_TFLITE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
)


OUTPUT_DIR = "./cropped_faces"
MODEL_DIR = "./tflite_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# API KEY
api_keys_str = os.getenv("API_KEYS", "")
api_keys = api_keys_str.split(",") if api_keys_str else []
use_api_keys = os.getenv("USE_API_KEYS", "False").lower() in ["true", "1", "yes"]


tflite_model_url = default_tflite_model_url
tflite_model_path = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")

try:
    urllib.request.urlretrieve(tflite_model_url, tflite_model_path)
    print(f"Model downloaded successfully to: {tflite_model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check the URL and your internet connection.")


def load_model():
    global classifier
    if classifier is None:
        logger.info("DEFAULT MODEL: " + default_model_name)
        classifier = pipeline(
            "image-classification",
            model=default_model_name,
            device=-1,
            accelerator="ort",
            token=access_token,
        )
        logger.info("Model loaded and cached")
    return classifier


def get_classifier():
    """Get the loaded classifier instance"""
    if classifier is None:
        raise RuntimeError("Model not loaded. Call load_model() first during startup.")
    return classifier
