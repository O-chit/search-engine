"""Lược đồ dữ liệu Pydantic cho API tìm kiếm và sức khỏe hệ thống."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SearchResultItem(BaseModel):
    """Mô hình dữ liệu của một kết quả tìm kiếm sản phẩm."""

    model_config = ConfigDict(extra="ignore")

    product_uid: str
    title: str
    brand: str
    snippet: str
    score: float
    position: int
    explored: bool = False
    debug: Optional[Dict[str, Any]] = None


class SearchTiming(BaseModel):
    """Mô hình dữ liệu ghi nhận thời gian xử lý từng công đoạn (mili-giây)."""

    parse: float
    retrieval: float
    rank: float
    total: float


class SearchResponse(BaseModel):
    """Mô hình phản hồi đầy đủ cho endpoint tìm kiếm."""

    model_config = ConfigDict(extra="ignore")

    query_id: str
    normalized_query: str
    lang: str
    personalized: bool = False
    degraded: bool = False
    total: int
    page: int
    size: int
    results: List[SearchResultItem]
    timing_ms: SearchTiming


class HealthResponse(BaseModel):
    """Mô hình phản hồi kiểm tra sức khỏe của dịch vụ và OpenSearch."""

    status: str
    opensearch: str
    model_loaded: bool = False
    document_count: int
