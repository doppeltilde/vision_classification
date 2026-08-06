import numpy as np
import mediapipe as mp
import logging
from PIL import Image, ImageDraw, ImageFont
from src.shared.shared import OUTPUT_DIR
import os
import uuid
import cv2

logger = logging.getLogger(__name__)

from src.shared.shared import get_model_by_name

hand_landmarker_model_path = get_model_by_name("Hand Landmarker")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
detector = HandLandmarker.create_from_options(options)


def mediapipe_hand_landmark_detection(
    img: Image.Image,
) -> tuple[bool, int, list, Image.Image]:
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = detector.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        if results.hand_landmarks:
            logger.info(len(results.hand_landmarks))
            hand_locations = []
            annotated_img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                handedness = results.handedness[idx][0]
                confidence_score = handedness.score
                category_name = handedness.category_name

                if confidence_score >= 0.5:
                    logger.info(confidence_score)

                    x_coordinates = [lm.x for lm in hand_landmarks]
                    y_coordinates = [lm.y for lm in hand_landmarks]
                    
                    x_min = int(min(x_coordinates) * img_width)
                    y_min = int(min(y_coordinates) * img_height)
                    x_max = int(max(x_coordinates) * img_width)
                    y_max = int(max(y_coordinates) * img_height)
                    width = x_max - x_min
                    height = y_max - y_min

                    hand_location = {
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
                        "landmarks": hand_landmarks,
                    }

                    hand_locations.append(hand_location)

                    points = [(int(lm.x * img_width), int(lm.y * img_height)) for lm in hand_landmarks]
                    for conn in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        pt1 = points[conn.start]
                        pt2 = points[conn.end]
                        cv2.line(annotated_img_cv, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

                    for pt in points:
                        cv2.circle(annotated_img_cv, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

            hand_locations.sort(key=lambda x: x["confidence"], reverse=True)
            annotated_img = Image.fromarray(cv2.cvtColor(annotated_img_cv, cv2.COLOR_BGR2RGB))

            hand_count = len(hand_locations)
            return hand_count > 0, hand_count, hand_locations, annotated_img
        else:
            return False, 0, [], img

    except Exception as e:
        logger.error(f"Error in hand landmark detection: {e}", exc_info=True)
        return False, 0, [], img


def save_annotated_hands(
    img: Image.Image,
    hand_locations: list[dict],
    file_id: str | None = None,
) -> str:
    if not file_id:
        file_id = uuid.uuid4().hex

    sorted_hands = sorted(hand_locations, key=lambda x: x["confidence"], reverse=True)

    annotated = img.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 20)
    except OSError:
        logger.warning("Failed to load font assets/Roboto-ExtraBold.ttf, using default.")
        font = ImageFont.load_default()

    img_width, img_height = img.size

    for hand in sorted_hands:
        coords = hand["pixel"]
        score = hand["confidence"]
        category = hand["category"]
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
            width=5,
        )

        text_y = max(5, ymin - 35)
        draw_overlay.text((xmin, text_y), score_text, font=font, fill="white")

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_hand.jpg")
    annotated.save(out_path, "JPEG", quality=95)
    logger.info(f"Saved single annotated image with {len(sorted_hands)} hand(s) -> {out_path}")

    return out_path