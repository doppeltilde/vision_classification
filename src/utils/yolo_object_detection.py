import os
import uuid
import logging
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from src.shared.shared import OUTPUT_DIR

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.2
EXCLUDED_CLASSES = ["make_love"]

model = YOLO("./yolo_models/yolo11-v1.1.pt")


def yolo_object_detection(
    img: Image.Image,
) -> tuple[bool, int, list, Image.Image]:
    try:
        img_rgb = np.array(img.convert("RGB"))
        img_bgr = img_rgb[..., ::-1]
        h, w = img_bgr.shape[:2]
        anchor = max(h, w)

        results = model(
            img_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=True,
        )

        result = results[0]
        detections = sv.Detections.from_ultralytics(result)

        class_names = [model.names[class_id] for class_id in detections.class_id]
        detections.data["class_name"] = np.array(class_names)

        if EXCLUDED_CLASSES and "class_name" in detections.data:
            mask = ~np.isin(detections.data["class_name"], EXCLUDED_CLASSES)
            detections = detections[mask]

        if len(detections) == 0:
            return False, 0, [], img

        enlarged_xyxy = detections.xyxy.copy()
        enlarged_xyxy[:, 0] -= 10
        enlarged_xyxy[:, 1] -= 10
        enlarged_xyxy[:, 2] += 10
        enlarged_xyxy[:, 3] += 10

        enlarged_xyxy[:, [0, 2]] = np.clip(enlarged_xyxy[:, [0, 2]], 0, w)
        enlarged_xyxy[:, [1, 3]] = np.clip(enlarged_xyxy[:, [1, 3]], 0, h)
        detections.xyxy = enlarged_xyxy

        object_locations = []
        for xyxy, conf, class_name in zip(
            detections.xyxy, detections.confidence, detections.data["class_name"]
        ):
            xmin, ymin, xmax, ymax = map(float, xyxy)
            box_w = xmax - xmin
            box_h = ymax - ymin

            object_locations.append(
                {
                    "normalized": {
                        "xmin": xmin / w,
                        "ymin": ymin / h,
                        "width": box_w / w,
                        "height": box_h / h,
                        "xmax": xmax / w,
                        "ymax": ymax / h,
                    },
                    "pixel": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "width": box_w,
                        "height": box_h,
                        "xmax": xmax,
                        "ymax": ymax,
                    },
                    "confidence": float(conf),
                    "category": str(class_name),
                }
            )

        object_locations.sort(key=lambda x: x["confidence"], reverse=True)

        pixelate_annotator = sv.PixelateAnnotator(pixel_size=int(anchor / 15))
        label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_position=sv.Position.CENTER,
            text_scale=max(0.4, anchor / 1700),
        )

        formatted_labels = [
            f"{name} {conf:.0%}"
            for name, conf in zip(detections.data["class_name"], detections.confidence)
        ]

        annotated_bgr = pixelate_annotator.annotate(
            scene=img_bgr.copy(), detections=detections
        )
        
        annotated_bgr = label_annotator.annotate(
            scene=annotated_bgr, detections=detections, labels=formatted_labels
        )

        annotated_img = Image.fromarray(annotated_bgr[..., ::-1])

        object_count = len(object_locations)
        print(object_count)
        return object_count > 0, object_count, object_locations, annotated_img

    except Exception as e:
        logger.error(f"Error in object detection: {e}", exc_info=True)
        return False, 0, [], img


def yolo_save_annotated_objects(
    img: Image.Image,
    object_locations: list[dict],
    file_id: str | None = None,
) -> str:
    if not file_id:
        file_id = uuid.uuid4().hex

    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_object.jpg")
    img.convert("RGB").save(out_path, "JPEG", quality=95)
    logger.info(
        f"Saved single annotated image with {len(object_locations)} object(s) -> {out_path}"
    )

    return out_path