from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
import fastapi_swagger_dark as fsd
import logging

from src.shared.shared import load_model
from src.api import classify
from src.api import classify_batch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    yield
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan, docs_url=None)
router = APIRouter()
fsd.install(router)
app.include_router(router)
app.include_router(classify.router)
app.include_router(classify_batch.router)


@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}
