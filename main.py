import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI

from src.api import classify, classify_batch, mediapipe_tasks
from src.helper.generate_api_key_and_hash import (
    generate_api_key_and_hash_with_salt,
)
from src.middleware.auth import get_api_key
from src.shared.shared import load_model, settings

logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Classify Images",
        "description": "Endpoints for image classification tasks.",
    },
    {
        "name": "Mediapipe Tasks",
        "description": "Endpoints for mediapipe vision tasks.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    numeric_log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_log_level)
    logger.info("Application starting up...")
    try:
        load_model()
        if logger.isEnabledFor(logging.DEBUG):
            generate_api_key_and_hash_with_salt()
        else:
            logger.info("Debug mode not enabled.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    yield
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan, openapi_tags=tags_metadata)

app.include_router(classify.router)
app.include_router(classify_batch.router)
app.include_router(mediapipe_tasks.router)


@app.get("/", dependencies=[Depends(get_api_key)])
def root():
    return {"res": "FastAPI is up and running!"}