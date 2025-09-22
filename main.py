from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends
import logging

from src.middleware.auth import get_api_key
from src.shared.shared import load_model, log_level
from src.api import classify
from src.api import classify_batch
from src.helper.generate_api_key_and_hash import generate_api_key_and_hash_with_salt

numeric_log_level = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=numeric_log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(lifespan=lifespan)
router = APIRouter()
app.include_router(router)
app.include_router(classify.router)
app.include_router(classify_batch.router)


@app.get("/", dependencies=[Depends(get_api_key)])
def root():
    return {"res": "FastAPI is up and running!"}
