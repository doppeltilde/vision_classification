import base64
import logging
import os
import uuid
from io import BytesIO
from typing import Optional

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

from src.shared.shared import OUTPUT_DIR, get_model_by_name

logger = logging.getLogger(__name__)

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

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (17, 19), (19, 21), (18, 20), (20, 22), (29, 31), (30, 32)
)

def mediapipe_pose_landmarker_detection(
    img: Image.Image,
    save_pose_landmark_file: bool = False,
) -> tuple[bool, int, list, str]:
    try:
        rgb_img = img.convert("RGB")
        img_array = np.array(rgb_img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        results = poselandmarker.detect(mp_image)

        pose_list = results.pose_landmarks if results.pose_landmarks else []
        num_detected_poses = len(pose_list)
        logger.info(f"Detected {num_detected_poses} pose(s)")

        if not pose_list:
            return False, 0, [], ""

        annotated_img = rgb_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        for pose_landmarks in pose_list:
            n_landmarks = len(pose_landmarks)
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < n_landmarks and end_idx < n_landmarks:
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    draw.line(
                        (
                            int(start.x * img_width),
                            int(start.y * img_height),
                            int(end.x * img_width),
                            int(end.y * img_height),
                        ),
                        fill="white",
                        width=3,
                    )

            for landmark in pose_landmarks:
                x = landmark.x * img_width
                y = landmark.y * img_height
                radius = 3.0
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=(144, 238, 144),
                    outline="white",
                )

        if save_pose_landmark_file:
            filename = f"{uuid.uuid4().hex}_pose_landmark.jpg"
            file_path = os.path.join(OUTPUT_DIR, filename)
            annotated_img.save(file_path, format="JPEG", quality=95)
            logger.info(f"Saved pose image to: {file_path}")

        buffered = BytesIO()
        annotated_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        annotated_base64 = f"data:image/png;base64,{img_base64}"

        return True, num_detected_poses, pose_list, annotated_base64

    except Exception as e:
        logger.error(f"Error in pose landmark: {e}", exc_info=True)
        return False, 0, [], ""