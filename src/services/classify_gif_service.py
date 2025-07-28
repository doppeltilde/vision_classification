from PIL import Image
from typing import Dict, Any
import logging, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from optimum.pipelines import pipeline

from src.services.classify_frame_service import classify_frame

logger = logging.getLogger(__name__)


async def process_animated_gif(
    contents: bytes,
    img: Image.Image,
    every_n_frame: int,
    score_threshold: float,
    max_workers: int,
    model: pipeline,
    label: str = "nsfw",
) -> Dict[str, Any]:
    """Process an animated GIF"""
    frame_count = img.n_frames
    frame_indices = list(range(0, frame_count, every_n_frame))

    logger.info(
        f"Processing GIF with {frame_count} frames, sampling every {every_n_frame} frames"
    )

    results = []
    stop_event = threading.Event()
    processed_frames = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_frame = {
            executor.submit(classify_frame, contents, idx, stop_event, model): idx
            for idx in frame_indices
        }

        # Process results as they complete
        for future in as_completed(future_to_frame):
            result = future.result()

            # Skip if marked as skipped
            if result.get("skipped"):
                continue

            results.append(result)
            processed_frames += 1

            # Check if should stop early
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
