import logging
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
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
    """
    Crop a single face from the image (with optional padding).
    Optionally saves the annotated crop and runs landmark detection on it.
    """
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
        score_text = f"Score: {confidence_score:.2f}"
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
    """
    Draw bounding boxes + confidence scores + light filter for ALL detected faces
    onto a single image and save it as one *_face.jpg.
    """
    if file_id is None:
        file_id = uuid.uuid4().hex

    # Convert base image to RGBA for alpha compositing
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
        score_text = f"Score: {score:.2f}"

        bbox = [(coords["xmin"], coords["ymin"]), (coords["xmax"], coords["ymax"])]

        # Draw semi-transparent white filter inside the box (RGBA white with 40/255 alpha)
        draw_overlay.rounded_rectangle(
            bbox,
            radius=15,
            fill=(255, 255, 255, 40),
            outline="white",
            width=5,
        )

        text_y = max(5, coords["ymin"] - 35)
        draw_overlay.text((coords["xmin"], text_y), score_text, font=font, fill="white")

    # Composite the transparent filter overlay onto the original image
    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_face.jpg")
    annotated.save(out_path, "JPEG", quality=95)
    logger.info(f"Saved single annotated image with {len(face_locations)} face(s) → {out_path}")

    if save_landmark:
        mediapipe_face_landmark_detection(annotated, f"{file_id}_full")

    return out_path