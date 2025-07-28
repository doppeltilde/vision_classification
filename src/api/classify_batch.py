from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List, Dict, Any

from src.middleware.auth import get_api_key
from src.api.classify import classify

router = APIRouter()


@router.post("/api/classify/batch", dependencies=[Depends(get_api_key)])
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
