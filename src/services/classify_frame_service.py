import threading

from PIL import Image
from typing import Dict, Any
import logging, io

from optimum.pipelines import pipeline

logger = logging.getLogger(__name__)


def classify_frame(
    img_bytes: bytes,
    frame_idx: int,
    stop_event: threading.Event,
    model: pipeline,
) -> Dict[str, Any]:
    if stop_event.is_set():
        return {"frame": frame_idx, "skipped": True}

    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img.seek(frame_idx)
            predictions = model(img)

            return {
                "frame": frame_idx,
                "predictions": predictions,
                "timestamp": frame_idx,
            }
    except Exception as e:
        logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
        return {"frame": frame_idx, "error": str(e)}
