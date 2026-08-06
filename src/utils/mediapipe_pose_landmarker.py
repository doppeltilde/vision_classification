import base64
import logging
import os
import uuid
from io import BytesIO
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from PIL import Image, ImageDraw, ImageOps

from src.shared.shared import OUTPUT_DIR, get_model_by_name

logger = logging.getLogger(__name__)

pose_landmarker_model_path = get_model_by_name("Pose Landmarker")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options_multi = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_landmarker_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
poselandmarker_multi = PoseLandmarker.create_from_options(options_multi)

options_single = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_landmarker_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
poselandmarker_single = PoseLandmarker.create_from_options(options_single)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (17, 19), (19, 21), (18, 20), (20, 22), (29, 31), (30, 32)
)

FACE_LANDMARK_INDICES = range(0, 11)
VISIBILITY_THRESHOLD = 0.5
CROP_MARGIN_RATIO = 0.15


def _pil_to_mp_image(pil_img: Image.Image) -> mp.Image:
    rgb_img = pil_img.convert("RGB")
    array = np.ascontiguousarray(np.array(rgb_img))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=array)


def _face_center_x(pose_landmarks, img_width: int) -> Optional[float]:
    xs = [
        pose_landmarks[i].x
        for i in FACE_LANDMARK_INDICES
        if i < len(pose_landmarks) and pose_landmarks[i].visibility > VISIBILITY_THRESHOLD
    ]
    if not xs:
        return None
    return (sum(xs) / len(xs)) * img_width


def _get_person_crop_boxes(pose_list, img_width: int, img_height: int):
    centers = [_face_center_x(p, img_width) for p in pose_list]

    if any(c is None for c in centers):
        return [(0, 0, img_width, img_height) for _ in pose_list]

    order = sorted(range(len(centers)), key=lambda i: centers[i])
    sorted_centers = [centers[i] for i in order]

    boundaries = [0.0]
    for i in range(len(sorted_centers) - 1):
        boundaries.append((sorted_centers[i] + sorted_centers[i + 1]) / 2)
    boundaries.append(float(img_width))

    crops = [None] * len(pose_list)
    for rank, idx in enumerate(order):
        left, right = boundaries[rank], boundaries[rank + 1]
        margin = (right - left) * CROP_MARGIN_RATIO
        left = max(0, left - margin)
        right = min(img_width, right + margin)
        crops[idx] = (int(left), 0, int(right), img_height)
    return crops


def _refine_pose_with_crop(rgb_img: Image.Image, crop_box, img_width, img_height):
    left, top, right, bottom = crop_box
    crop_w, crop_h = right - left, bottom - top
    if crop_w <= 0 or crop_h <= 0:
        return None

    crop_img = rgb_img.crop(crop_box)
    mp_crop = _pil_to_mp_image(crop_img)
    result = poselandmarker_single.detect(mp_crop)

    if not result.pose_landmarks:
        return None

    refined = result.pose_landmarks[0]
    remapped = []
    for lm in refined:
        full_x = (left + lm.x * crop_w) / img_width
        full_y = (top + lm.y * crop_h) / img_height
        remapped.append(
            NormalizedLandmark(
                x=full_x,
                y=full_y,
                z=lm.z,
                visibility=lm.visibility,
                presence=lm.presence,
            )
        )
    return remapped


def mediapipe_pose_landmarker_detection(
    img: Image.Image,
    save_pose_landmark_file: bool = False,
) -> tuple[bool, int, list, str]:
    try:
        img = ImageOps.exif_transpose(img)
        rgb_img = img.convert("RGB")
        img_width, img_height = rgb_img.size

        mp_image = _pil_to_mp_image(rgb_img)
        results = poselandmarker_multi.detect(mp_image)

        pose_list = results.pose_landmarks if results.pose_landmarks else []
        num_detected_poses = len(pose_list)
        logger.info(f"Detected {num_detected_poses} pose(s)")

        if not pose_list:
            return False, 0, [], ""

        crop_boxes = _get_person_crop_boxes(pose_list, img_width, img_height)
        refined_pose_list = []
        for i, pose_landmarks in enumerate(pose_list):
            refined = _refine_pose_with_crop(
                rgb_img, crop_boxes[i], img_width, img_height
            )
            refined_pose_list.append(refined if refined is not None else pose_landmarks)
        pose_list = refined_pose_list

        annotated_img = rgb_img.copy()
        overlay = Image.new("RGBA", annotated_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for pose_landmarks in pose_list:
            n_landmarks = len(pose_landmarks)

            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < n_landmarks and end_idx < n_landmarks:
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]

                    if (
                        start.visibility < VISIBILITY_THRESHOLD
                        or end.visibility < VISIBILITY_THRESHOLD
                    ):
                        continue

                    x1, y1 = start.x * img_width, start.y * img_height
                    x2, y2 = end.x * img_width, end.y * img_height

                    draw.line(
                        (x1, y1, x2, y2),
                        fill=(255, 255, 255, 50),
                        width=7,
                        joint="curve",
                    )
                    draw.line(
                        (x1, y1, x2, y2),
                        fill=(255, 255, 255, 220),
                        width=3,
                        joint="curve",
                    )

            for landmark in pose_landmarks:
                if landmark.visibility < VISIBILITY_THRESHOLD:
                    continue

                x = landmark.x * img_width
                y = landmark.y * img_height

                r_outer = 5.0
                draw.ellipse(
                    (x - r_outer, y - r_outer, x + r_outer, y + r_outer),
                    fill=(255, 255, 255, 60),
                )
                r_inner = 3.5
                draw.ellipse(
                    (x - r_inner, y - r_inner, x + r_inner, y + r_inner),
                    outline=(255, 255, 255, 255),
                    width=2,
                )

        annotated_img = Image.alpha_composite(
            annotated_img.convert("RGBA"), overlay
        ).convert("RGB")

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