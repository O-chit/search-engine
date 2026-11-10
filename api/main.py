"""Điểm khởi chạy ứng dụng FastAPI cho hệ thống tìm kiếm."""

from contextlib import asynccontextmanager
import json
import logging
import os
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from api.routes.events import router as events_router
from api.routes.health import router as health_router
from api.routes.search import router as search_router
from api.routes.session import router as session_router
from config import API_HOST, API_PORT, EMBEDDINGS_NPY_PATH, UID_INDEX_PATH
from indexing.model import get_embedding_model
from storage.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo tài nguyên ứng dụng khi khởi động máy chủ."""
    logger.info("Khởi tạo cơ sở dữ liệu SQLite...")
    init_db()

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
app.include_router(events_router)
app.include_router(session_router)

# Đảm bảo khởi tạo các bảng khi nạp ứng dụng
init_db()

if os.path.exists("web"):
    app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/", response_class=FileResponse)
def serve_index() -> FileResponse:
    """Phục vụ giao diện tìm kiếm người dùng web/index.html."""
    return FileResponse("web/index.html")


def main() -> None:
    """Chạy máy chủ Uvicorn phục vụ API tìm kiếm."""
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
