import logging
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import mediapipe as mp
from src.shared.shared import OUTPUT_DIR
from src.utils.mediapipe_face_landmark import mediapipe_face_landmark_detection

logger = logging.getLogger(__name__)


def crop_face_from_image(
    img: Image.Image,
    face_location: dict,
    padding: float = 0.1,
    save_cropped: bool = False,
    save_landmark: bool = False,
    file_id: str | None = None,
) -> Image.Image:
    if file_id is None:
        file_id = uuid.uuid4().hex

    img_width, img_height = img.size
    pixel_coords = face_location["pixel"]
    confidence_score = face_location["confidence"]

    width_padding = int(pixel_coords["width"] * padding)
    height_padding = int(pixel_coords["height"] * padding)

    left = max(0, pixel_coords["xmin"] - width_padding)
    top = max(0, pixel_coords["ymin"] - height_padding)
    right = min(img_width, pixel_coords["xmax"] + width_padding)
    bottom = min(img_height, pixel_coords["ymax"] + height_padding)

    clean_face_img = img.crop((left, top, right, bottom))

    if save_cropped:
        score_text = f"Score: {confidence_score * 100:.0f}%"
        try:
            font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 14)
        except OSError:
            logger.warning("Failed to load font assets/Roboto-ExtraBold.ttf, using default.")
            font = ImageFont.load_default()

        annotated_crop = clean_face_img.copy()
        draw_crop = ImageDraw.Draw(annotated_crop)
        draw_crop.text((5, 5), score_text, font=font, fill="white")

        cropped_path = os.path.join(OUTPUT_DIR, f"{file_id}_cropped.jpg")
        annotated_crop.save(cropped_path, "JPEG", quality=95)
        logger.info(f"Saved cropped face with score to: {cropped_path}")

        if save_landmark:
            mediapipe_face_landmark_detection(annotated_crop, f"{file_id}_cropped")

    return clean_face_img


def save_annotated_faces(
    img: Image.Image,
    face_locations: list[dict],
    file_id: str | None = None,
    save_landmark: bool = False,
) -> str:
    if not file_id:
        file_id = uuid.uuid4().hex

    annotated = img.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 20)
    except OSError:
        logger.warning("Failed to load font assets/Roboto-ExtraBold.ttf, using default.")
        font = ImageFont.load_default()

    for face in face_locations:
        coords = face["pixel"]
        score = face["confidence"]
        score_text = f"Score: {score * 100:.0f}%"


        bbox = [(coords["xmin"], coords["ymin"]), (coords["xmax"], coords["ymax"])]

        draw_overlay.rounded_rectangle(
            bbox,
            radius=15,
            fill=(255, 255, 255, 40),
            outline="white",
            width=5,
        )

        text_y = max(5, coords["ymin"] - 35)
        draw_overlay.text((coords["xmin"], text_y), score_text, font=font, fill="white")

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_face.jpg")
    annotated.save(out_path, "JPEG", quality=95)
    logger.info(f"Saved single annotated image with {len(face_locations)} face(s) → {out_path}")

    if save_landmark:
        full_landmark_img = img.copy()
        img_width, img_height = img.size

        for face in face_locations:
            pixel_coords = face["pixel"]
            width_padding = int(pixel_coords["width"] * 0.1)
            height_padding = int(pixel_coords["height"] * 0.1)

            left = max(0, pixel_coords["xmin"] - width_padding)
            top = max(0, pixel_coords["ymin"] - height_padding)
            right = min(img_width, pixel_coords["xmax"] + width_padding)
            bottom = min(img_height, pixel_coords["ymax"] + height_padding)

            cropped_face = img.crop((left, top, right, bottom))
            success, _, landmarks = mediapipe_face_landmark_detection(cropped_face, file_id)

            if success and landmarks:
                crop_np = np.array(cropped_face.convert('RGB'))
                h_c, w_c, _ = crop_np.shape
                annotated_crop_cv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

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

                for face_landmarks in landmarks:
                    points = [(int(lm.x * w_c), int(lm.y * h_c)) for lm in face_landmarks]
                    for conn in CONNECTIONS:
                        pt1 = points[conn.start]
                        pt2 = points[conn.end]
                        cv2.line(annotated_crop_cv, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

                    key_indices = [1, 33, 61, 199, 263, 291, 362, 454, 234]
                    for idx in key_indices:
                        if idx < len(points):
                            cv2.circle(annotated_crop_cv, points[idx], 3, (255, 255, 255), -1, cv2.LINE_AA)

                processed_crop_pil = Image.fromarray(cv2.cvtColor(annotated_crop_cv, cv2.COLOR_BGR2RGB))
                full_landmark_img.paste(processed_crop_pil, (left, top))

        landmark_path = os.path.join(OUTPUT_DIR, f"{file_id}_full_landmark.jpg")
        full_landmark_img.save(landmark_path, "JPEG", quality=95)
        logger.info(f"Saved composite landmark full image to: {landmark_path}")

    return out_path