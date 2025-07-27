from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from PIL import Image, UnidentifiedImageError
import io, os, uuid, filetype, threading, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from optimum.pipelines import pipeline

from src.middleware.auth import get_api_key
from src.shared.shared import access_token, default_model_name

import numpy as np
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./cropped_faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

mp_face_detection = mp.solutions.face_detection
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
    label: str = None,
    detect_faces: bool = False,
    save_cropped: bool = False,
) -> Dict[str, Any]:
    """
    Classify images or animated GIFs for NSFW content

    Args:
        file: Image file to classify
        every_n_frame: Process every Nth frame for GIFs (default: 3)
        score_threshold: Threshold for early stopping on positive detection
        max_workers: Maximum number of worker threads for GIF processing
        detect_faces: Whether to perform face detection before classification (default: False)
        save_cropped: Whether to save cropped faces (default: False)

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
            return await process_single_image(
                img,
                classifier,
                detect_faces,
                save_cropped=save_cropped,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_single_image(
    img: Image.Image,
    model: pipeline,
    detect_faces: bool = False,
    save_cropped: bool = False,
) -> Dict[str, Any]:
    try:
        start_time = time.time()

        if detect_faces:
            faces_detected, face_count, face_locations = detect_faces_in_image(img)

            if not faces_detected:
                end_time = time.time()
                processing_time = end_time - start_time

                return {
                    "type": "single_image",
                    "faces_detected": False,
                    "face_count": 0,
                    "predictions": None,
                    "face_locations": [],
                    "processing_time": processing_time,
                }
            predictions_cropped = []
            for face_location in face_locations:
                cropped_face = crop_face_from_image(
                    img,
                    face_location,
                    save_cropped=save_cropped,
                )
                prediction_cropped = model(cropped_face)
                predictions_cropped.append(prediction_cropped)

            end_time = time.time()
            processing_time = end_time - start_time

            return {
                "type": "multi_face",
                "faces_detected": True,
                "face_count": face_count,
                "face_locations": face_locations,
                "predictions": predictions_cropped,
                "processing_time": processing_time,
            }
        else:
            predictions = model(img)

            end_time = time.time()
            processing_time = end_time - start_time

            return {
                "type": "single_image",
                "faces_detected": False,
                "face_count": 0,
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


def detect_faces_in_image(
    img: Image.Image,
) -> tuple[bool, int, list]:
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        with mp_face_detection.FaceDetection(
            model_selection=0,
        ) as face_detection:

            results = face_detection.process(img_array)

            logger.info(results.detections)

            if results.detections:
                face_locations = []
                for detection in results.detections:
                    confidence_score = detection.score[0]
                    if confidence_score >= 0.55:
                        bbox = detection.location_data.relative_bounding_box
                        logger.info(confidence_score)

                        x_min = int(bbox.xmin * img_width)
                        y_min = int(bbox.ymin * img_height)
                        width = int(bbox.width * img_width)
                        height = int(bbox.height * img_height)
                        x_max = x_min + width
                        y_max = y_min + height

                        face_location = {
                            "normalized": {
                                "xmin": bbox.xmin,
                                "ymin": bbox.ymin,
                                "width": bbox.width,
                                "height": bbox.height,
                                "xmax": bbox.xmin + bbox.width,
                                "ymax": bbox.ymin + bbox.height,
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
                        }

                        face_locations.append(face_location)

                face_count = len(results.detections)
                return True, face_count, face_locations
            else:
                return False, 0, []

    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return False, 0, []


def crop_face_from_image(
    img: Image.Image,
    face_location: dict,
    padding: float = 0.1,
    save_cropped: bool = False,
) -> Image.Image:
    img_width, img_height = img.size

    pixel_coords = face_location["pixel"]

    width_padding = int(pixel_coords["width"] * padding)
    height_padding = int(pixel_coords["height"] * padding)

    left = max(0, pixel_coords["xmin"] - width_padding)
    top = max(0, pixel_coords["ymin"] - height_padding)
    right = min(img_width, pixel_coords["xmax"] + width_padding)
    bottom = min(img_height, pixel_coords["ymax"] + height_padding)

    face_img = img.crop((left, top, right, bottom))

    if save_cropped:
        filename = f"face_{uuid.uuid4().hex[:8]}.jpg"
        file_path = os.path.join(OUTPUT_DIR, filename)

        face_img.save(file_path, "JPEG", quality=95)
        logger.info(f"Saved cropped face to: OUTPUT_DIR")

    return face_img
