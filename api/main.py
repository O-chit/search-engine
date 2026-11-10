"""Điểm khởi chạy ứng dụng FastAPI cho hệ thống tìm kiếm."""

from contextlib import asynccontextmanager
import json
import logging
import os
import random
import time
from typing import AsyncIterator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from api.routes.events import router as events_router
from api.routes.health import router as health_router
from api.routes.search import router as search_router
from api.routes.session import router as session_router
from config import (
    API_HOST,
    API_PORT,
    RETRIEVAL_K,
    SPLIT_PATH,
    WARMUP_QUERY_COUNT,
    WARMUP_SEED,
)
from indexing.model import get_embedding_model, get_product_embeddings, get_uid_index
from query.parser import parse
from retrieval.bm25 import search_bm25
from retrieval.vector import search_vector
from storage.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo tài nguyên ứng dụng khi khởi động máy chủ."""
    logger.info("Khởi tạo cơ sở dữ liệu SQLite...")
    init_db()

    logger.info("Nạp mô hình embedding...")
    model = get_embedding_model()
    _ = model.encode(["warmup"], normalize_embeddings=True)

    logger.info("Làm nóng OpenSearch tổng quát với %d truy vấn từ train...", WARMUP_QUERY_COUNT)
    warmup_queries: List[str] = []
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_queries = split_data.get("train", [])
        if train_queries:
            rng = random.Random(WARMUP_SEED)
            warmup_queries = rng.sample(train_queries, min(WARMUP_QUERY_COUNT, len(train_queries)))

    t_warmup_start = time.perf_counter()
    first_ms = 0.0
    last_ms = 0.0
    for idx, q in enumerate(warmup_queries):
        t_q_start = time.perf_counter()
        wp = parse(q, with_vector=True)
        await search_bm25(wp, k=RETRIEVAL_K)
        await search_vector(wp, k=RETRIEVAL_K)
        q_ms = (time.perf_counter() - t_q_start) * 1000.0
        if idx == 0:
            first_ms = q_ms
        if idx == len(warmup_queries) - 1:
            last_ms = q_ms

    total_warmup_ms = (time.perf_counter() - t_warmup_start) * 1000.0
    warmup_log = {
        "warmup_queries_count": len(warmup_queries),
        "total_warmup_ms": round(total_warmup_ms, 2),
        "first_query_ms": round(first_ms, 2),
        "last_query_ms": round(last_ms, 2),
    }
    logger.info(json.dumps(warmup_log, ensure_ascii=False))

    embeddings = get_product_embeddings()
    uid_index = get_uid_index()

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
