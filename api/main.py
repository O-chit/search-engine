"""Điểm khởi chạy ứng dụng FastAPI cho hệ thống tìm kiếm."""

from contextlib import asynccontextmanager
import json
import logging
import os
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from api.routes.health import router as health_router
from api.routes.search import router as search_router
from config import API_HOST, API_PORT, EMBEDDINGS_NPY_PATH, UID_INDEX_PATH
from indexing.model import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo tài nguyên ứng dụng khi khởi động máy chủ."""
    logger.info("Nạp mô hình embedding và làm nóng...")
    model = get_embedding_model()
    _ = model.encode(["warmup query"], normalize_embeddings=True)

    embeddings = None
    uid_index = None
    if os.path.exists(EMBEDDINGS_NPY_PATH):
        embeddings = np.load(EMBEDDINGS_NPY_PATH, mmap_mode="r")
    if os.path.exists(UID_INDEX_PATH):
        with open(UID_INDEX_PATH, "r", encoding="utf-8") as f:
            uid_index = json.load(f)

    app.state.embedding_model = model
    app.state.embeddings = embeddings
    app.state.uid_index = uid_index
    app.state.model_loaded = True
    logger.info("Hoàn tất nạp mô hình embedding và trạng thái ứng dụng.")
    yield


app = FastAPI(title="Search Engine Hybrid API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(search_router)


def main() -> None:
    """Chạy máy chủ Uvicorn phục vụ API tìm kiếm."""
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
