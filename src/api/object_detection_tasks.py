from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from PIL import Image, UnidentifiedImageError
import io, puremagic, os, hmac, hashlib, time
from typing import Dict, Any
import logging, base64
from fastapi.responses import FileResponse
from pathlib import Path
from src.shared.shared import OUTPUT_DIR
from src.shared.shared import settings

from src.middleware.auth import get_api_key
from src.utils.yolo_object_detection import (
    yolo_object_detection,
    yolo_save_annotated_objects,
)
from src.utils.litert_objection_detection import (
    litert_object_detection,
    litert_save_annotated_objects,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Object Detection Tasks"])

URL_EXPIRATION_SECONDS = 600


def generate_presigned_url(filename: str) -> str:
    expires = int(time.time()) + URL_EXPIRATION_SECONDS
    message = f"{filename}:{expires}".encode()
    signature = hmac.new(
        settings.image_signing_secret.encode(), message, hashlib.sha256
    ).hexdigest()
    return f"/images/{filename}?expires={expires}&signature={signature}"


@router.get(
    "/images/{filename}",
    dependencies=[Depends(get_api_key)],
)
async def serve_protected_image(
    filename: str,
    expires: int = Query(...),
    signature: str = Query(...),
):
    if int(time.time()) > expires:
        raise HTTPException(status_code=400, detail="Image URL has expired")

    message = f"{filename}:{expires}".encode()
    expected_signature = hmac.new(
        settings.image_signing_secret.encode(), message, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    safe_name = Path(filename).name
    file_path = Path(OUTPUT_DIR) / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
    )


@router.post(
    "/api/yolo/sensitive_content_object_detection", dependencies=[Depends(get_api_key)]
)
async def classify(
    file: UploadFile = File(...),
    apply_face_blackbox: bool = True,
    apply_pixelation: bool = True,
    show_labels: bool = True,
    return_image_url: bool = False,
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

    objects_detected, objects_count, objects_locations, image = yolo_object_detection(
        img,
        apply_face_blackbox=apply_face_blackbox,
        apply_pixelation=apply_pixelation,
        show_labels=show_labels,
    )

    if objects_detected and objects_count > 0:
        saved_path = yolo_save_annotated_objects(
            image,
            objects_locations,
            file_id=file_id,
        )
        image_filename = Path(saved_path).name
    else:
        image_filename = f"{file_id}.jpg"

    image_url = generate_presigned_url(image_filename)

    response = {
        "objects_detected": objects_detected,
        "objects_count": objects_count,
        "objects_locations": objects_locations,
    }

    if return_image_url:
        response["image_url"] = image_url

    return response


@router.post(
    "/api/litert/sensitive_content_object_detection",
    dependencies=[Depends(get_api_key)],
)
async def classify(
    file: UploadFile = File(...),
    apply_face_blackbox: bool = True,
    apply_pixelation: bool = True,
    show_labels: bool = True,
    return_image_url: bool = False,
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

    objects_detected, objects_count, objects_locations, image = litert_object_detection(
        img,
        apply_face_blackbox=apply_face_blackbox,
        apply_pixelation=apply_pixelation,
        show_labels=show_labels,
    )

    if objects_detected and objects_count > 0:
        saved_path = litert_save_annotated_objects(
            image,
            objects_locations,
            file_id=file_id,
        )
        image_filename = Path(saved_path).name
    else:
        image_filename = f"{file_id}.jpg"

    image_url = generate_presigned_url(image_filename)

    response = {
        "objects_detected": objects_detected,
        "objects_count": objects_count,
        "objects_locations": objects_locations,
    }

    if return_image_url:
        response["image_url"] = image_url

    return response
