import logging
import os
import uuid
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

from src.shared.shared import OUTPUT_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = "./models/model.litert"
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.30

NAMES = ["nahf", "oerngf", "crnav", "erne", "ingvan"]

CT = str.maketrans(
    "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

def get_name(text: str) -> str:
    return text.translate(CT)

CLASS_NAMES = [get_name(name) for name in NAMES]

COLORS = [
    (90, 140, 195),  # Light Brown
    (128, 0, 128),  # Purple
    (0, 0, 255),  # Red
    (90, 140, 195),  # Light Brown
    (180, 105, 255),  # Pink
]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, INPUT_H, INPUT_W, _ = input_details[0]["shape"]


def pixelate_roi(target_img, src_img, x1, y1, x2, y2, pixel_size):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return target_img

    roi = src_img[y1:y2, x1:x2]
    grid_w = max(1, w // pixel_size)
    grid_h = max(1, h // pixel_size)

    small = cv2.resize(roi, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    target_img[y1:y2, x1:x2] = pixelated
    return target_img


def litert_object_detection(
    img: Image.Image,
) -> tuple[bool, int, list, Image.Image]:
    try:
        img_rgb = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]
        anchor = max(orig_h, orig_w)

        resized = cv2.cvtColor(
            cv2.resize(img_bgr, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB
        )
        input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details[0]["index"])[0]
        if raw_output.shape[0] < raw_output.shape[1]:
            raw_output = raw_output.T

        num_cols = raw_output.shape[1]
        boxes_to_process = []

        if num_cols == 6:
            for pred in raw_output:
                x1, y1, x2, y2, score, class_id = pred
                if score < CONF_THRESHOLD:
                    continue

                if x2 <= 1.0 and y2 <= 1.0:
                    x1, y1, x2, y2 = (
                        x1 * orig_w,
                        y1 * orig_h,
                        x2 * orig_w,
                        y2 * orig_h,
                    )
                else:
                    x1, y1, x2, y2 = (
                        x1 * (orig_w / INPUT_W),
                        y1 * (orig_h / INPUT_H),
                        x2 * (orig_w / INPUT_W),
                        y2 * (orig_h / INPUT_H),
                    )

                boxes_to_process.append(
                    (x1, y1, x2, y2, float(score), int(class_id))
                )

        else:
            boxes_cxcywh = raw_output[:, :4]
            class_scores = raw_output[:, 4:]

            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            mask = confidences > CONF_THRESHOLD
            boxes_cxcywh = boxes_cxcywh[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]

            if len(confidences) > 0:
                boxes_nms = []
                for cx, cy, w, h in boxes_cxcywh:
                    if cx <= 1.0 and cy <= 1.0:
                        cx, cy, w, h = (
                            cx * INPUT_W,
                            cy * INPUT_H,
                            w * INPUT_W,
                            h * INPUT_H,
                        )

                    x_min = int((cx - (w / 2.0)) * (orig_w / INPUT_W))
                    y_min = int((cy - (h / 2.0)) * (orig_h / INPUT_H))
                    box_w = int(w * (orig_w / INPUT_W))
                    box_h = int(h * (orig_h / INPUT_H))
                    boxes_nms.append([x_min, y_min, box_w, box_h])

                indices = cv2.dnn.NMSBoxes(
                    boxes_nms,
                    confidences.tolist(),
                    CONF_THRESHOLD,
                    NMS_THRESHOLD,
                )

                if len(indices) > 0:
                    indices = (
                        indices.flatten()
                        if isinstance(indices, np.ndarray)
                        else indices
                    )
                    for idx in indices:
                        x, y, w, h = boxes_nms[idx]
                        boxes_to_process.append(
                            (
                                x,
                                y,
                                x + w,
                                y + h,
                                float(confidences[idx]),
                                int(class_ids[idx]),
                            )
                        )

        if len(boxes_to_process) == 0:
            return False, 0, [], img

        object_locations = []
        annotated_bgr = img_bgr.copy()
        pixel_size = max(1, int(anchor / 15))

        for x1, y1, x2, y2, conf, c_id in boxes_to_process:
            c_name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"class_{c_id}"

            x1_pad = int(np.clip(x1 - 10, 0, orig_w))
            y1_pad = int(np.clip(y1 - 10, 0, orig_h))
            x2_pad = int(np.clip(x2 + 10, 0, orig_w))
            y2_pad = int(np.clip(y2 + 10, 0, orig_h))

            box_w = x2_pad - x1_pad
            box_h = y2_pad - y1_pad

            object_locations.append(
                {
                    "normalized": {
                        "xmin": x1_pad / orig_w,
                        "ymin": y1_pad / orig_h,
                        "width": box_w / orig_w,
                        "height": box_h / orig_h,
                        "xmax": x2_pad / orig_w,
                        "ymax": y2_pad / orig_h,
                    },
                    "pixel": {
                        "xmin": x1_pad,
                        "ymin": y1_pad,
                        "width": box_w,
                        "height": box_h,
                        "xmax": x2_pad,
                        "ymax": y2_pad,
                    },
                    "confidence": float(conf),
                    "category": str(c_name),
                }
            )

            annotated_bgr = pixelate_roi(
                annotated_bgr, img_bgr, x1_pad, y1_pad, x2_pad, y2_pad, pixel_size
            )

            color = COLORS[c_id % len(COLORS)]
            text = f"{c_name} {conf:.0%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, anchor / 1700)
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            center_x = (x1_pad + x2_pad) // 2
            center_y = (y1_pad + y2_pad) // 2
            text_x = max(0, center_x - (text_w // 2))
            text_y = max(text_h, center_y + (text_h // 2))

            cv2.rectangle(
                annotated_bgr,
                (text_x - 4, text_y - text_h - 4),
                (text_x + text_w + 4, text_y + baseline + 2),
                color,
                -1,
            )
            cv2.putText(
                annotated_bgr,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        object_locations.sort(key=lambda x: x["confidence"], reverse=True)

        annotated_img = Image.fromarray(
            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        )
        object_count = len(object_locations)
        print(object_count)

        return object_count > 0, object_count, object_locations, annotated_img

    except Exception as e:
        logger.error(f"Error in object detection: {e}", exc_info=True)
        return False, 0, [], img


def litert_save_annotated_objects(
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