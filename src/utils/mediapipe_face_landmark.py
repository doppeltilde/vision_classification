import numpy as np
import mediapipe as mp
import logging, os

logger = logging.getLogger(__name__)

from PIL import Image, ImageDraw

from src.shared.shared import OUTPUT_DIR
from src.shared.shared import get_model_by_name

face_detection_model_path = get_model_by_name("Face Landmark")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_detection_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=5,
)
landmarker = FaceLandmarker.create_from_options(options)


def mediapipe_face_landmark_detection(
    img: Image.Image,
    fileId: str = None,
) -> tuple[bool, int, list]:
    try:
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = landmarker.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        num_faces = len(results.face_landmarks) if results.face_landmarks else 0
        logger.info(f"Detected {num_faces} face(s)")

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                for landmark in face_landmarks:
                    x = landmark.x * img_width
                    y = landmark.y * img_height
                    radius = 0.5
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill="red",
                        outline="red",
                    )
        filename = f"{fileId}_landmark.jpg"
        file_path = os.path.join(OUTPUT_DIR, filename)
        annotated_img.save(file_path)
        logger.info(f"Saved full image with score to: {file_path}")

        return False, 0, []

    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return False, 0, []
