from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image, UnidentifiedImageError
import io, puremagic, os, base64
from typing import Dict, Any, Optional
import logging

from src.middleware.auth import get_api_key
from src.shared.crop_face_from_image import crop_face_from_image, save_annotated_faces
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
    
    file_id = os.path.splitext(file.filename)[0] if file.filename else "uploaded_file"

    try:
        mime_type = puremagic.from_string(contents, mime=True)
        is_image = mime_type.startswith("image/")
    except puremagic.PureError:
        is_image = False

    if not is_image:
        raise HTTPException(
            status_code=400, detail="File is not a supported image type"
        )

    try:
        image_stream = io.BytesIO(contents)
        img = Image.open(image_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    faces_detected, face_count, face_locations = mediapipe_face_detection(img)
    
    if faces_detected and face_count > 0:
        save_annotated_faces(
            img,
            face_locations,
            file_id=file_id,
            save_landmark=save_landmark_file,
        )

    for face_location in face_locations:
        crop_face_from_image(
            img,
            face_location,
            save_cropped=save_cropped_file,
            save_landmark=False,
            file_id=file_id,
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

    try:
        mime_type = puremagic.from_string(contents, mime=True)
        is_image = mime_type.startswith("image/")
    except puremagic.PureError:
        is_image = False

    if not is_image:
        raise HTTPException(
            status_code=400, detail="File is not a supported image type"
        )

    try:
        image_stream = io.BytesIO(contents)
        img = Image.open(image_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    print(save_pose_landmark_file)
    pose_detected, pose_count, pose_locations, base64img = (
        mediapipe_pose_landmarker_detection(
            img, save_pose_landmark_file=save_pose_landmark_file
        )
    )

    return {
        "pose_detected": pose_detected,
        "pose_count": pose_count,
        "pose_locations": pose_locations,
    }


@router.post("/api/mediapipe/image_classification", dependencies=[Depends(get_api_key)])
async def classify(
    file: UploadFile = File(...),
    load_from_local_storage: bool = False,
    local_model_path: str = None,
) -> Dict[str, Any]:
    contents = await file.read()

    try:
        mime_type = puremagic.from_string(contents, mime=True)
        is_image = mime_type.startswith("image/")
    except puremagic.PureError:
        is_image = False

    if not is_image:
        raise HTTPException(
            status_code=400, detail="File is not a supported image type"
        )

    try:
        image_stream = io.BytesIO(contents)
        img = Image.open(image_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    classification, count, locations, image = mediapipe_image_classification(
        img,
        load_from_local_storage=load_from_local_storage,
        local_model_path=local_model_path,
    )

    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "classification": classification,
        "count": count,
        "locations": locations,
        "image": image_base64,
    }