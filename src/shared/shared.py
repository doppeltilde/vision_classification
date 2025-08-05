from dotenv import load_dotenv
import os, urllib.request
from optimum.pipelines import pipeline
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv(
    "DEFAULT_MODEL_NAME", "onnx-community/nsfw_image_detection-ONNX"
)


OUTPUT_DIR = "./cropped_faces"
MODEL_DIR = "./mediapipe_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# API KEY
stored_api_key_salt = os.getenv("API_KEY_SALT", "")
stored_api_key_hash = os.getenv("API_KEY_HASH", "")
use_api_key = os.getenv("USE_API_KEY", "False").lower() in ["true", "1", "yes"]

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# START MEDIAPIPE
# https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
# https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
# https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf

mediapipe_model_storage_url = "https://storage.googleapis.com/mediapipe-models"

models = {
    "Face Detection": {
        "env_var": "DEFAULT_FACE_DETECTION_MODEL_URL",
        "default_path": "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "filename": "blaze_face_short_range.tflite",
    },
    "Face Landmark": {
        "env_var": "DEFAULT_FACE_LANDMARK_MODEL_URL",
        "default_path": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "filename": "face_landmarker.task",
    },
    "Gesture Recognition": {
        "env_var": "DEFAULT_GESTURE_RECOGNITION_MODEL_URL",
        "default_path": "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        "filename": "gesture_recognizer.task",
    },
    "Object Detection": {
        "env_var": "DEFAULT_OBJECT_DETECTION_MODEL_URL",
        "default_path": "object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
        "filename": "efficientdet_lite0.tflite",
    },
}


def get_model_by_name(model_name: str) -> str:
    return os.path.join(MODEL_DIR, models[model_name]["filename"])


for model_name, config in models.items():
    model_url = os.getenv(
        config["env_var"], f"{mediapipe_model_storage_url}/{config['default_path']}"
    )
    model_path = os.path.join(MODEL_DIR, config["filename"])

    if os.path.exists(model_path):
        logger.info(
            f"{model_name} model already exists at: {model_path}. Skipping download."
        )
        continue

    try:
        urllib.request.urlretrieve(model_url, model_path)
        logger.info(f"{model_name} model downloaded successfully to: {model_path}")
    except Exception as e:
        logger.error(f"Error downloading {model_name} model: {e}")
        logger.error(f"URL tried: {model_url}")

# END MEDIAPIPE

_model_cache: Dict[str, Any] = {}


def load_model(model_name: Optional[str] = None):
    try:
        model_to_load = model_name or default_model_name
        logger.debug("DEFAULT MODEL: " + model_to_load)

        if model_to_load in _model_cache:
            logger.debug(f"Model {model_to_load} already loaded, using cached version")
            return _model_cache[model_to_load]

        classifier = pipeline(
            "image-classification",
            model=model_to_load,
            device=-1,
            accelerator="ort",
            token=access_token,
        )
        _model_cache[model_to_load] = classifier
        logger.info("Model loaded and cached")
        return classifier
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise
