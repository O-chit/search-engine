"""Quản lý kết nối và các thao tác cơ bản với OpenSearch."""

import json
import logging
from typing import Any, Dict

from opensearchpy import OpenSearch

from config import OPENSEARCH_URL

logger = logging.getLogger(__name__)

_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    """Tạo hoặc trả về phiên bản OpenSearch client dùng chung."""
    global _client
    if _client is None:
        _client = OpenSearch(
            hosts=[OPENSEARCH_URL],
            use_ssl=False,
            verify_certs=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            maxsize=10,
        )
    return _client


def check_cluster_health() -> Dict[str, Any]:
    """Lấy thông tin trạng thái cụm OpenSearch."""
    client = get_opensearch_client()
    return client.cluster.health()


def get_document_count(index_name: str) -> int:
    """Đếm tổng số tài liệu hiện có trong một chỉ mục OpenSearch."""
    client = get_opensearch_client()
    if not client.indices.exists(index=index_name):
        return 0
    res = client.count(index=index_name)
    return int(res.get("count", 0))


def create_index(index_name: str, mapping_path: str, recreate: bool = False) -> None:
    """Tạo chỉ mục OpenSearch từ file mapping cấu hình, hỗ trợ xoá tạo lại nếu recreate=True."""
    client = get_opensearch_client()
    if client.indices.exists(index=index_name):
        if recreate:
            logger.info("Xoá chỉ mục cũ: %s", index_name)
            client.indices.delete(index=index_name)
        else:
            logger.info("Chỉ mục %s đã tồn tại, bỏ qua tạo mới.", index_name)
            return
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_body = json.load(f)
    client.indices.create(index=index_name, body=mapping_body)
    logger.info("Đã tạo thành công chỉ mục: %s", index_name)
