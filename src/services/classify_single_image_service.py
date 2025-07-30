from PIL import Image
from fastapi import HTTPException
from typing import Dict, Any
import logging, time

from optimum.pipelines import pipeline

from src.shared.crop_face_from_image import crop_face_from_image
from src.utils.mediapipe_face_detector import mediapipe_face_detection

logger = logging.getLogger(__name__)


async def process_single_image(
    img: Image.Image,
    model: pipeline,
    detect_faces: bool = False,
    save_cropped: bool = False,
    save_landmark: bool = False,
    return_face_locations: bool = False,
) -> Dict[str, Any]:
    try:
        start_time = time.time()

        if detect_faces:
            faces_detected, face_count, face_locations = mediapipe_face_detection(img)

            if not faces_detected:
                end_time = time.time()
                processing_time = end_time - start_time

                result = {
                    "type": "single_image",
                    "faces_detected": False,
                    "face_count": 0,
                    "predictions": None,
                    "processing_time": processing_time,
                }

                if return_face_locations:
                    result["face_locations"] = []

                return result

            predictions_cropped = []
            for face_location in face_locations:
                cropped_face = crop_face_from_image(
                    img,
                    face_location,
                    save_cropped=save_cropped,
                    save_landmark=save_landmark,
                )
                prediction_cropped = model(cropped_face)
                predictions_cropped.extend(prediction_cropped)

            end_time = time.time()
            processing_time = end_time - start_time

            result = {
                "type": "multi_face",
                "faces_detected": True,
                "face_count": face_count,
                "predictions": predictions_cropped,
                "processing_time": processing_time,
            }

            if return_face_locations:
                result["face_locations"] = face_locations

            return result
        else:
            predictions = model(img)

            end_time = time.time()
            processing_time = end_time - start_time

            return {
                "type": "single_image",
                "faces_detected": False,
                "face_count": 0,
                "predictions": predictions,
                "processing_time": processing_time,
            }
    except Exception as e:
        logger.error(f"Error processing single image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")
    finally:
        img.close()
        del img
