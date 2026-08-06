import logging
import os
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from optimum.pipelines import pipeline
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MEDIAPIPE_MODEL_STORAGE_URL = "https://storage.googleapis.com/mediapipe-models"


class Settings(BaseSettings):
    access_token: Optional[str] = None
    default_model_name: str = "onnx-community/nsfw_image_detection-ONNX"
    api_key_salt: str = ""
    api_key_hash: str = ""
    use_api_key: bool = False
    log_level: str = "INFO"

    default_image_classification_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"
    )
    default_face_detection_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
    )
    default_face_landmark_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    default_gesture_recognition_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    )
    default_object_detection_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
    )
    default_pose_landmarker_model_url: str = Field(
        default=f"{MEDIAPIPE_MODEL_STORAGE_URL}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

OUTPUT_DIR = "./cropped_faces"
MODEL_DIR = "./mediapipe_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def get_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    return os.path.basename(path)


models = {
    "Image Classification": {
        "url": settings.default_image_classification_model_url,
    },
    "Face Detection": {
        "url": settings.default_face_detection_model_url,
    },
    "Face Landmark": {
        "url": settings.default_face_landmark_model_url,
    },
    "Gesture Recognition": {
        "url": settings.default_gesture_recognition_model_url,
    },
    "Object Detection": {
        "url": settings.default_object_detection_model_url,
    },
    "Pose Landmarker": {
        "url": settings.default_pose_landmarker_model_url,
        "model_card": "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf",
    },
}

for config in models.values():
    config["filename"] = get_filename_from_url(config["url"])


def get_custom_model(model_name: str) -> str:
    return os.path.join(MODEL_DIR, model_name)


def get_model_by_name(model_name: str) -> str:
    return os.path.join(MODEL_DIR, models[model_name]["filename"])


for model_name, config in models.items():
    model_url = config["url"]
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

_model_cache: Dict[str, Any] = {}


def load_model(model_name: Optional[str] = None):
    try:
        model_to_load = model_name or settings.default_model_name
        logger.debug("DEFAULT MODEL: " + model_to_load)

        if model_to_load in _model_cache:
            logger.debug(f"Model {model_to_load} already loaded, using cached version")
            return _model_cache[model_to_load]

        classifier = pipeline(
            "image-classification",
            model=model_to_load,
            device=-1,
            accelerator="ort",
            token=settings.access_token,
        )
        _model_cache[model_to_load] = classifier
        logger.info("Model loaded and cached")
        return classifier
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise