"""Điểm khởi chạy ứng dụng FastAPI cho hệ thống tìm kiếm."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.search import router as search_router
from config import API_HOST, API_PORT

app = FastAPI(title="Search Engine BM25 API")

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
