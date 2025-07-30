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


OUTPUT_DIR = "./cropped_faces"
MODEL_DIR = "./mediapipe_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# API KEY
api_keys_str = os.getenv("API_KEYS", "")
api_keys = api_keys_str.split(",") if api_keys_str else []
use_api_keys = os.getenv("USE_API_KEYS", "False").lower() in ["true", "1", "yes"]


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
    if classifier is None:
        raise RuntimeError("Model not loaded. Call load_model() first during startup.")
    return classifier
