from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.middleware.auth import get_api_key

router = APIRouter()


@router.post("/api/classify-video", dependencies=[Depends(get_api_key)])
async def classify_video(
    file: UploadFile = File(...),
    every_n_frame: int = 3,
    score_threshold: float = 0.7,
    max_workers: int = None,
    label: str = "nsfw",
):
    try:
        return "hello"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
