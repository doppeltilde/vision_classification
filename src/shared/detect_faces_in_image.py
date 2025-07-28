import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging

logger = logging.getLogger(__name__)

from PIL import Image

from src.shared.shared import tflite_model_path

base_options = python.BaseOptions(model_asset_path=tflite_model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def detect_faces_in_image(
    img: Image.Image,
) -> tuple[bool, int, list]:
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = detector.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        if results.detections:
            face_locations = []
            for detection in results.detections:
                confidence_score = detection.categories[0].score
                if confidence_score >= 0.5:
                    logger.info(confidence_score)

                    bbox = detection.bounding_box

                    x_min = bbox.origin_x
                    y_min = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    x_max = x_min + width
                    y_max = y_min + height

                    face_location = {
                        "normalized": {
                            "xmin": x_min / img_width,
                            "ymin": y_min / img_height,
                            "width": width / img_width,
                            "height": height / img_height,
                            "xmax": x_max / img_width,
                            "ymax": y_max / img_height,
                        },
                        "pixel": {
                            "xmin": x_min,
                            "ymin": y_min,
                            "width": width,
                            "height": height,
                            "xmax": x_max,
                            "ymax": y_max,
                        },
                        "confidence": confidence_score,
                    }

                    face_locations.append(face_location)

            face_count = len(results.detections)
            return True, face_count, face_locations
        else:
            return False, 0, []

    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return False, 0, []
