from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image, UnidentifiedImageError
import io, filetype
from typing import Dict, Any
import logging

from src.middleware.auth import get_api_key
from src.services.classify_single_image_service import process_single_image
from src.services.classify_gif_service import process_animated_gif
from src.shared.shared import get_classifier


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/classify", dependencies=[Depends(get_api_key)])
async def classify(
    file: UploadFile = File(...),
    every_n_frame: int = 3,
    score_threshold: float = 0.7,
    max_workers: int = None,
    label: str = None,
    detect_faces: bool = False,
    save_cropped: bool = False,
    save_landmark: bool = False,
    return_face_locations: bool = False,
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
        save_landmark: Whether to save face landmarks (default: False)
        return_face_locations: Whether to return face locations (default: False)

    Returns:
        Classification results
    """
    if get_classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate parameters
    if every_n_frame < 1:
        raise HTTPException(status_code=400, detail="every_n_frame must be >= 1")
    if not (0.0 <= score_threshold <= 1.0):
        raise HTTPException(
            status_code=400, detail="score_threshold must be between 0.0 and 1.0"
        )

    try:
        classifier = get_classifier()

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
        if (img.format == "GIF" or img.format == "WEBP") and getattr(
            img, "is_animated", False
        ):
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
                save_landmark=save_landmark,
                return_face_locations=return_face_locations,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
