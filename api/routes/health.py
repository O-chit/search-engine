"""Endpoint kiểm tra trạng thái sức khỏe của dịch vụ."""

import logging
from fastapi import APIRouter, Request

from api.schemas import HealthResponse
from config import INDEX_NAME
from retrieval.client import check_cluster_health, get_document_count

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Trả về trạng thái hoạt động của OpenSearch, mô hình và số lượng tài liệu."""
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    try:
        cluster_info = check_cluster_health()
        os_status = str(cluster_info.get("status", "unknown"))
        doc_count = get_document_count(INDEX_NAME)
        return HealthResponse(
            status="ok",
            opensearch=os_status,
            model_loaded=model_loaded,
            document_count=doc_count,
        )
    except Exception as exc:
        logger.error("Lỗi khi kiểm tra sức khỏe OpenSearch: %s", exc, exc_info=True)
        return HealthResponse(
            status="degraded",
            opensearch="unreachable",
            model_loaded=model_loaded,
            document_count=0,
        )
