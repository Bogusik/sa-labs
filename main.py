import logging
from fastapi import FastAPI

from src.routes import router

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)
