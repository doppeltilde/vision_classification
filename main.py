from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from PIL import Image, UnidentifiedImageError
import io
import filetype
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any, Optional
import logging
import time
from optimum.pipelines import pipeline

from src.middleware.auth import get_api_key
from src.shared.shared import access_token, default_model_name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

classifier: Optional[pipeline] = None


@app.on_event("startup")
async def load_model():
    """Load the classification model at startup"""
    global classifier
    try:
        logger.info("DEFAULT MODEL: " + default_model_name)

        classifier = pipeline(
            "image-classification",
            model=default_model_name,
            device=-1,
            accelerator="ort",
            token=access_token,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}


def classify_frame(
    img_bytes: bytes, frame_idx: int, stop_event: threading.Event, model: pipeline
) -> Dict[str, Any]:
    """
    Classify a single frame from an animated GIF

    Args:
        img_bytes: Raw image bytes
        frame_idx: Frame index to process
        stop_event: Threading event to signal early termination
        model: The loaded classification model

    Returns:
        Dictionary with frame index and predictions or error
    """
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


@app.post("/api/classify", dependencies=[Depends(get_api_key)])
async def classify(
    file: UploadFile = File(...),
    every_n_frame: int = 3,
    score_threshold: float = 0.7,
    max_workers: int = None,
    label: str = "nsfw",
) -> Dict[str, Any]:
    """
    Classify images or animated GIFs for NSFW content

    Args:
        file: Image file to classify
        every_n_frame: Process every Nth frame for GIFs (default: 3)
        score_threshold: Threshold for early stopping on positive detection
        max_workers: Maximum number of worker threads for GIF processing

    Returns:
        Classification results
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate parameters
    if every_n_frame < 1:
        raise HTTPException(status_code=400, detail="every_n_frame must be >= 1")
    if not (0.0 <= score_threshold <= 1.0):
        raise HTTPException(
            status_code=400, detail="score_threshold must be between 0.0 and 1.0"
        )

    try:
        contents = await file.read()

        # Validate file type
        if not filetype.is_image(contents):
            raise HTTPException(
                status_code=400, detail="File is not a supported image type"
            )

        # Open image
        try:
            image_stream = io.BytesIO(contents)
            img = Image.open(image_stream)
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400, detail="Invalid or corrupted image file"
            )

        # Process based on image type
        if img.format == "GIF" and getattr(img, "is_animated", False):
            return await process_animated_gif(
                contents,
                img,
                every_n_frame,
                score_threshold,
                max_workers,
                classifier,
                label,
            )
        else:
            return await process_single_image(img, classifier)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_single_image(
    img: Image.Image,
    model: pipeline,
) -> Dict[str, Any]:
    """Process a single image"""
    try:
        start_time = time.time()

        predictions = model(img)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "type": "single_image",
            "predictions": predictions,
            "processing_time": processing_time,
        }
    except Exception as e:
        logger.error(f"Error processing single image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")


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


@app.post("/api/classify/batch", dependencies=[Depends(get_api_key)])
async def classify_batch(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Classify multiple images at once"""
    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 files allowed per batch"
        )

    results = []
    for i, file in enumerate(files):
        try:
            result = await classify(file)
            results.append({"file_index": i, "filename": file.filename, **result})
        except Exception as e:
            results.append(
                {"file_index": i, "filename": file.filename, "error": str(e)}
            )

    return {"batch_results": results}
