import numpy as np
import mediapipe as mp
import logging
from PIL import Image, ImageDraw, ImageFont
from src.shared.shared import OUTPUT_DIR
import os
import uuid

logger = logging.getLogger(__name__)

from src.shared.shared import get_model_by_name

object_detection_model_path = get_model_by_name("Object Detection")

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=object_detection_model_path),
    running_mode=VisionRunningMode.IMAGE,
)
detector = ObjectDetector.create_from_options(options)


def mediapipe_object_detection(
    img: Image.Image,
) -> tuple[bool, int, list, Image.Image]:
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = detector.detect(mp_image)

        logger.info(f"Detection Results: {results}")

        if results.detections:
            object_locations = []
            for detection in results.detections:
                category = detection.categories[0]
                confidence_score = category.score
                category_name = category.category_name
                
                if confidence_score >= 0.5:
                    logger.info(confidence_score)

                    bbox = detection.bounding_box

                    x_min = bbox.origin_x
                    y_min = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    x_max = x_min + width
                    y_max = y_min + height

                    object_location = {
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
                    }

                    object_locations.append(object_location)

            object_locations.sort(key=lambda x: x["confidence"], reverse=True)

            annotated_img = img.copy()

            object_count = len(object_locations)
            return object_count > 0, object_count, object_locations, annotated_img
        else:
            return False, 0, [], img

    except Exception as e:
        logger.error(f"Error in object detection: {e}", exc_info=True)
        return False, 0, [], img

logger = logging.getLogger(__name__)

def save_annotated_objects(
    img: Image.Image,
    object_locations: list[dict],
    file_id: str | None = None,
) -> str:
    if not file_id:
        file_id = uuid.uuid4().hex

    sorted_objects = sorted(object_locations, key=lambda x: x["confidence"], reverse=True)

    annotated = img.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 20)
    except OSError:
        logger.warning("Failed to load font assets/Roboto-ExtraBold.ttf, using default.")
        font = ImageFont.load_default()

    for obj in sorted_objects:
        coords = obj["pixel"]
        score = obj["confidence"]
        category = obj["category"]
        score_text = f"{category}: {score * 100:.0f}%"

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

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_object.jpg")
    annotated.save(out_path, "JPEG", quality=95)
    logger.info(f"Saved single annotated image with {len(sorted_objects)} object(s) -> {out_path}")

    return out_path