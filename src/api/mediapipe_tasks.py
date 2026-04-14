from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image, UnidentifiedImageError
import io, filetype
from typing import Dict, Any, Optional
import logging

from src.middleware.auth import get_api_key
from src.shared.crop_face_from_image import crop_face_from_image
from src.utils.mediapipe_face_detector import mediapipe_face_detection
from src.utils.mediapipe_pose_landmarker import mediapipe_pose_landmarker_detection
from src.utils.mediapipe_image_classification import mediapipe_image_classification

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Mediapipe Tasks"])


@router.post("/api/mediapipe/face_detection", dependencies=[Depends(get_api_key)])
async def classify(
    file: UploadFile = File(...),
    save_cropped_file: bool = False,
    save_landmark_file: bool = False,
) -> Dict[str, Any]:
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
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    faces_detected, face_count, face_locations = mediapipe_face_detection(img)
    for face_location in face_locations:
        cropped_face = crop_face_from_image(
            img,
            face_location,
            save_cropped=save_cropped_file,
            save_landmark=save_landmark_file,
        )
    return {
        "faces_detected": faces_detected,
        "face_count": face_count,
        "face_locations": face_locations,
    }


@router.post(
    "/api/mediapipe/pose_landmark_detection", dependencies=[Depends(get_api_key)]
)
async def classify(
    file: UploadFile = File(...),
    save_pose_landmark_file: bool = False,
) -> Dict[str, Any]:
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
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    pose_detected, pose_count, pose_locations, base64img = (
        mediapipe_pose_landmarker_detection(
            img, save_pose_landmark_file=save_pose_landmark_file
        )
    )

    return {
        "pose_detected": pose_detected,
        "pose_count": pose_count,
        # "base64_image": base64img,
        "pose_locations": pose_locations,
    }


@router.post("/api/mediapipe/image_classification", dependencies=[Depends(get_api_key)])
async def classify(
    file: UploadFile = File(...),
    load_from_local_storage: bool = False,
    local_model_path: str = None,
) -> Dict[str, Any]:
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
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    classification, count, locations = mediapipe_image_classification(
        img,
        load_from_local_storage=load_from_local_storage,
        local_model_path=local_model_path,
    )

    return {
        "classification": classification,
        "count": count,
        # "base64_image": base64img,
        "locations": locations,
    }
