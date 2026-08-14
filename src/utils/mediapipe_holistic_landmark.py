import logging
import os
import uuid
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.shared.shared import OUTPUT_DIR, get_model_by_name

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face connections
CONNECTIONS = (
    list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
    + list(mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
)

# Pose connections
POSE_CONNECTIONS = list(mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS)

holistic_model_path = get_model_by_name("Holistic Landmarker")


def mediapipe_holistic_landmark_detection(
    img: Image.Image,
) -> tuple[bool, int, list, Image.Image]:
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=holistic_model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_mask=False,
    )

    try:
        with HolisticLandmarker.create_from_options(options) as landmarker:
            img_array = np.array(img)
            img_height, img_width = img_array.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
            results = landmarker.detect(mp_image)

            logger.info(f"Detection Results: {results}")

            holistic_locations = []
            annotated_img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            landmarks_to_process = [
                ("Pose", results.pose_landmarks),
                ("Face", results.face_landmarks),
                ("Left Hand", results.left_hand_landmarks),
                ("Right Hand", results.right_hand_landmarks),
            ]

            for category_name, landmark_list in landmarks_to_process:
                if landmark_list:
                    confidence_score = 1.0

                    x_coordinates = [lm.x for lm in landmark_list]
                    y_coordinates = [lm.y for lm in landmark_list]

                    x_min = int(min(x_coordinates) * img_width)
                    y_min = int(min(y_coordinates) * img_height)
                    x_max = int(max(x_coordinates) * img_width)
                    y_max = int(max(y_coordinates) * img_height)
                    width = x_max - x_min
                    height = y_max - y_min

                    holistic_location = {
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
                        "category": category_name,
                        "landmarks": landmark_list,
                    }

                    holistic_locations.append(holistic_location)

                    points = [
                        (int(lm.x * img_width), int(lm.y * img_height))
                        for lm in landmark_list
                    ]

                    if category_name == "Face":
                        for connection in CONNECTIONS:
                            start_idx = connection.start
                            end_idx = connection.end
                            if start_idx < len(points) and end_idx < len(points):
                                cv2.line(
                                    annotated_img_cv,
                                    points[start_idx],
                                    points[end_idx],
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                    elif category_name == "Pose":
                        # Draw pose skeleton lines
                        for connection in POSE_CONNECTIONS:
                            start_idx = connection.start
                            end_idx = connection.end
                            if start_idx < len(points) and end_idx < len(points):
                                cv2.line(
                                    annotated_img_cv,
                                    points[start_idx],
                                    points[end_idx],
                                    (255, 255, 255),
                                    2,  # slightly thicker for pose
                                    cv2.LINE_AA,
                                )
                        # Also draw the pose keypoints
                        for pt in points:
                            cv2.circle(
                                annotated_img_cv,
                                pt,
                                3,
                                (255, 255, 255),
                                -1,
                                cv2.LINE_AA,
                            )
                    else:
                        # Hands – points only
                        for pt in points:
                            cv2.circle(
                                annotated_img_cv,
                                pt,
                                2,
                                (255, 255, 255),
                                -1,
                                cv2.LINE_AA,
                            )

            if holistic_locations:
                holistic_locations.sort(key=lambda x: x["confidence"], reverse=True)
                annotated_img = Image.fromarray(
                    cv2.cvtColor(annotated_img_cv, cv2.COLOR_BGR2RGB)
                )
                detected_count = len(holistic_locations)
                return True, detected_count, holistic_locations, annotated_img
            else:
                return False, 0, [], img

    except Exception as e:
        logger.error(f"Error in holistic landmark detection: {e}", exc_info=True)
        return False, 0, [], img


def save_annotated_holistic(
    img: Image.Image,
    holistic_locations: list[dict],
    file_id: str | None = None,
) -> str:
    if not file_id:
        file_id = uuid.uuid4().hex

    sorted_locations = sorted(
        holistic_locations, key=lambda x: x["confidence"], reverse=True
    )

    annotated = img.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 20)
    except OSError:
        logger.warning(
            "Failed to load font assets/Roboto-ExtraBold.ttf, using default."
        )
        font = ImageFont.load_default()

    img_width, img_height = img.size

    for loc in sorted_locations:
        coords = loc["pixel"]
        score = loc["confidence"]
        category = loc["category"]
        score_text = f"{category}: {score * 100:.0f}%"

        width_padding = int(coords["width"] * 0.10)
        height_padding = int(coords["height"] * 0.10)

        xmin = max(0, coords["xmin"] - width_padding)
        ymin = max(0, coords["ymin"] - height_padding)
        xmax = min(img_width, coords["xmax"] + width_padding)
        ymax = min(img_height, coords["ymax"] + height_padding)

        bbox = [(xmin, ymin), (xmax, ymax)]

        draw_overlay.rounded_rectangle(
            bbox,
            radius=15,
            fill=(255, 255, 255, 40),
            outline="white",
            width=3,
        )

        text_y = max(5, ymin - 35)
        draw_overlay.text(
            (xmin, text_y),
            score_text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_holistic.jpg")
    annotated.save(out_path, "JPEG", quality=95)
    logger.info(
        f"Saved single annotated image with {len(sorted_locations)} component(s) -> {out_path}"
    )

    return out_path
