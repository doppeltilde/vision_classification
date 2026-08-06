import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional
from PIL import Image
from fastapi import HTTPException
from src.shared.crop_face_from_image import crop_face_from_image, save_annotated_faces
from src.utils.mediapipe_face_detector import mediapipe_face_detection
from src.shared.shared import load_model

logger = logging.getLogger(__name__)


async def process_single_image(
    img: Image.Image,
    detect_faces: bool = False,
    save_cropped: bool = False,
    save_landmark: bool = False,
    return_face_locations: bool = False,
    model_to_load: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    try:
        model = await asyncio.to_thread(load_model, model_to_load)

        if not detect_faces:
            predictions = await asyncio.to_thread(model, img)
            result = {
                "type": "single_image",
                "faces_detected": False,
                "face_count": 0,
                "predictions": predictions,
            }
        else:
            faces_detected, face_count, face_locations = await asyncio.to_thread(
                mediapipe_face_detection, img
            )

            if not faces_detected:
                result = {
                    "type": "single_image",
                    "faces_detected": False,
                    "face_count": 0,
                    "predictions": None,
                }
            else:
                # One shared ID for the whole image
                file_id = uuid.uuid4().hex

                # Save a SINGLE annotated full image with ALL faces
                if save_cropped:
                    save_annotated_faces(
                        img,
                        face_locations,
                        file_id=file_id,
                        save_landmark=save_landmark,
                    )

                predictions_cropped = []
                for i, loc in enumerate(face_locations):
                    cropped = crop_face_from_image(
                        img,
                        loc,
                        save_cropped=save_cropped,          # individual crops (optional)
                        save_landmark=save_landmark,
                        file_id=f"{file_id}_{i}",           # unique name per crop
                    )
                    pred = await asyncio.to_thread(model, cropped)
                    if pred:
                        predictions_cropped.extend(pred)

                result = {
                    "type": "multi_face",
                    "faces_detected": True,
                    "face_count": face_count,
                    "predictions": predictions_cropped,
                }

            if return_face_locations:
                result["face_locations"] = face_locations if faces_detected else []

        result["processing_time"] = time.perf_counter() - start_time
        return result

    except Exception as e:
        logger.error(f"Error processing single image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")