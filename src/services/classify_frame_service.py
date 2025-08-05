import threading

from PIL import Image
from typing import Dict, Any, Optional
import logging, io

from src.shared.shared import load_model


logger = logging.getLogger(__name__)


def classify_frame(
    img_bytes: bytes,
    frame_idx: int,
    stop_event: threading.Event,
    model_to_load: Optional[str] = None,
) -> Dict[str, Any]:
    if stop_event.is_set():
        return {"frame": frame_idx, "skipped": True}

    try:
        model = load_model(model_to_load)

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
