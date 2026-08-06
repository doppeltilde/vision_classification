import numpy as np
import mediapipe as mp
import logging, os
from typing import Optional
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

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
    min_face_detection_confidence=0.3,
)
landmarker = FaceLandmarker.create_from_options(options)

CONNECTIONS = (
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW) +
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
)

def mediapipe_face_landmark_detection(
    img: Image.Image,
    fileId: Optional[str] = None,
) -> tuple[bool, int, list]:
    try:
        img_array = np.array(img.convert('RGB'))
        h, w, _ = img_array.shape
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        results = landmarker.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        num_faces = len(results.face_landmarks) if results.face_landmarks else 0
        logger.info(f"Detected {num_faces} face(s)")

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

                for conn in CONNECTIONS:
                    pt1 = points[conn.start]
                    pt2 = points[conn.end]
                    cv2.line(annotated_img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

                key_indices = [1, 33, 61, 199, 263, 291, 362, 454, 234]
                for idx in key_indices:
                    if idx < len(points):
                        cv2.circle(annotated_img, points[idx], 3, (255, 255, 255), -1, cv2.LINE_AA)

        final_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        filename = f"{fileId}_landmark.jpg"
        file_path = os.path.join(OUTPUT_DIR, filename)
        final_img.save(file_path)
        logger.info(f"Saved full image with score to: {file_path}")

        return True, num_faces, results.face_landmarks

    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return False, 0, []