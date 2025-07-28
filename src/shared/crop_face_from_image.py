import os

from PIL import Image, ImageDraw, ImageFont
import uuid
import logging

logger = logging.getLogger(__name__)

from src.shared.shared import OUTPUT_DIR


def crop_face_from_image(
    img: Image.Image,
    face_location: dict,
    padding: float = 0.1,
    save_cropped: bool = False,
) -> Image.Image:
    img_width, img_height = img.size

    pixel_coords = face_location["pixel"]
    confidence_score = face_location["confidence"]

    width_padding = int(pixel_coords["width"] * padding)
    height_padding = int(pixel_coords["height"] * padding)

    left = max(0, pixel_coords["xmin"] - width_padding)
    top = max(0, pixel_coords["ymin"] - height_padding)
    right = min(img_width, pixel_coords["xmax"] + width_padding)
    bottom = min(img_height, pixel_coords["ymax"] + height_padding)

    face_img = img.crop((left, top, right, bottom))

    if save_cropped:
        score_text = f"Score: {confidence_score:.2f}"

        text_x = 5
        text_y = 5

        font_path = "assets/Roboto-ExtraBold.ttf"
        font_size = 30

        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Successfully loaded font: {font_path} with size {font_size}")

        except IOError:
            font = ImageFont.load_default()

        annotated_img = img.copy()
        drawFull = ImageDraw.Draw(annotated_img)

        pixel_coords = face_location["pixel"]

        x1 = pixel_coords["xmin"]
        y1 = pixel_coords["ymin"]
        x2 = pixel_coords["xmax"]
        y2 = pixel_coords["ymax"]

        drawFull.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
        drawFull.text((text_x, text_y), score_text, font=font, fill="red")

        fileId = uuid.uuid4().hex

        filename = f"face_{fileId}.jpg"
        file_path = os.path.join(OUTPUT_DIR, filename)
        annotated_img.save(file_path, "JPEG", quality=95)
        logger.info(f"Saved full image with score to: {file_path}")

        draw = ImageDraw.Draw(face_img)
        draw.text((text_x, text_y), score_text, font=font, fill="red")

        croppedfilename = f"cropped_face_{fileId}.jpg"
        cropped_file_path = os.path.join(OUTPUT_DIR, croppedfilename)
        face_img.save(cropped_file_path, "JPEG", quality=95)
        logger.info(f"Saved cropped face with score to: {cropped_file_path}")

    return face_img
