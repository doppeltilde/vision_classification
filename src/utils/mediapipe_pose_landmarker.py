import numpy as np
import mediapipe as mp
import logging, os
from typing import Optional
import uuid
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

from PIL import Image, ImageDraw

from src.shared.shared import OUTPUT_DIR
from src.shared.shared import get_model_by_name

pose_landmarker_model_path = get_model_by_name("Pose Landmarker")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_landmarker_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=5,
)
poselandmarker = PoseLandmarker.create_from_options(options)

def mediapipe_pose_landmarker_detection(
    img: Image.Image,
    save_pose_landmark_file: bool = False,
) -> tuple[bool, int, list, str]:
    try:
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = poselandmarker.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        pose_list = results.pose_landmarks if results.pose_landmarks else []
        num_detected_poses = len(pose_list)
        logger.info(f"Detected {num_detected_poses} pose(s)")

        if num_detected_poses > 0:
            POSE_CONNECTIONS = [
                # Face
                (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
                # Torso
                (11,12),(11,23),(12,24),(23,24),
                # Left arm
                (11,13),(13,15),(15,17),(15,19),(15,21),
                # Right arm
                (12,14),(14,16),(16,18),(16,20),(16,22),
                # Left leg
                (23,25),(25,27),(27,29),(27,31),
                # Right leg
                (24,26),(26,28),(28,30),(28,32),
                # Hands & feet
                (17,19),(19,21),(18,20),(20,22),(29,31),(30,32)
            ]

            for idx, pose_landmarks in enumerate(pose_list):
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        start = pose_landmarks[start_idx]
                        end = pose_landmarks[end_idx]

                        x1 = int(start.x * img_width)
                        y1 = int(start.y * img_height)
                        x2 = int(end.x * img_width)
                        y2 = int(end.y * img_height)

                        draw.line((x1, y1, x2, y2), fill="white", width=3)

                for landmark in pose_landmarks:
                    x = landmark.x * img_width
                    y = landmark.y * img_height
                    radius = 3.0
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(144, 238, 144),
                        outline="white"
                    )
            filename = f"{str(uuid.uuid4())}_pose_landmark.jpg"
            file_path = os.path.join(OUTPUT_DIR, filename)
            if save_pose_landmark_file:
                annotated_img.save(file_path)
                logger.info(f"Saved full image with score to: {file_path}")

            buffered = BytesIO()
            annotated_img.save(buffered, format="PNG", optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            annotated_base64 = f"data:image/png;base64,{img_base64}"
            return True, num_detected_poses, pose_list, annotated_base64
        
        else:
            return False, 0, [], ""

    except Exception as e:
        logger.error(f"Error in pose landmark: {e}", exc_info=True)
        return False, 0, [], ""
