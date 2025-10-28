from PIL import Image
from typing import Dict, Any, Optional
import logging, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import HTTPException

from src.services.classify_frame_service import classify_frame

logger = logging.getLogger(__name__)


async def process_animated_gif(
    contents: bytes,
    img: Image.Image,
    every_n_frame: int,
    score_threshold: float,
    max_workers: Optional[int] = None,
    label: Optional[str] = None,
    model_to_load: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        frame_count = getattr(img, "n_frames", 1)
        frame_indices = list(range(0, frame_count, every_n_frame))

        logger.info(
            f"Processing GIF with {frame_count} frames, sampling every {every_n_frame} frames"
        )

        results = []
        stop_event = threading.Event()
        processed_frames = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_frame = {
                executor.submit(
                    classify_frame,
                    contents,
                    idx,
                    stop_event,
                    model_to_load,
                ): idx
                for idx in frame_indices
            }

            for future in as_completed(future_to_frame):
                result = future.result()

                if result.get("skipped"):
                    continue

                results.append(result)
                processed_frames += 1

                if "predictions" in result and result["predictions"]:
                    for pred in result["predictions"]:
                        if (
                            pred["label"].lower() == label
                            and pred["score"] > score_threshold
                        ):
                            logger.info(
                                f"Early stop triggered at frame {result['frame']} with label {pred['label']} score {pred['score']}"
                            )
                            stop_event.set()
                            break

                if stop_event.is_set():
                    break

        results.sort(key=lambda x: x.get("frame", 0))
        total_frames_processed = len([r for r in results if "predictions" in r])

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "type": "animated_gif",
            "total_frames": frame_count,
            "processed_frames": total_frames_processed,
            "sampled_every_n": every_n_frame,
            "early_stopped": stop_event.is_set(),
            "frame_results": results,
            "processing_time": processing_time,
        }
    except Exception as e:
        logger.error(f"Error processing animated GIF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")
    finally:
        img.close()
        del img
