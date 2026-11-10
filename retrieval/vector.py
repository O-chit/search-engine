"""Truy xuất tài liệu bằng thuật toán Vector qua OpenSearch."""

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple

from config import BRANCH_TIMEOUT_MS, INDEX_NAME, RETRIEVAL_K
from query.parser import ParsedQuery, encode_query
from retrieval.client import get_opensearch_client

logger = logging.getLogger(__name__)


class VectorRetrievalError(Exception):
    """Lỗi xảy ra trong quá trình truy xuất Vector từ OpenSearch."""

    pass


@dataclass
class VectorHit:
    """Cấu trúc dữ liệu kết quả trả về từ nhánh Vector."""

    product_uid: str
    score: float
    rank: int
    title: str = ""
    brand: str = ""
    bullets: str = ""
    description: str = ""


def _execute_vector_search(
    query: ParsedQuery,
    index_name: str,
    k: int,
) -> List[VectorHit]:
    """Thực thi truy vấn OpenSearch Vector (k-NN) đồng bộ."""
    client = get_opensearch_client()
    query_vector = query.vector
    if query_vector is None:
        query_vector, _ = encode_query(query.normalized)

    query_body: Dict[str, Any] = {
        "size": k,
        "timeout": f"{BRANCH_TIMEOUT_MS}ms",
        "_source": ["product_uid", "title", "brand", "bullets", "description"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    try:
        response = client.search(
            body=query_body,
            index=index_name,
            request_timeout=(BRANCH_TIMEOUT_MS / 1000.0) + 1.0,
        )
    except Exception as exc:
        logger.error("Lỗi khi gọi tìm kiếm Vector trên OpenSearch: %s", exc, exc_info=True)
        raise VectorRetrievalError(f"OpenSearch vector search failed: {exc}") from exc

    hits = response.get("hits", {}).get("hits", [])
    results: List[VectorHit] = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append(
            VectorHit(
                product_uid=str(source.get("product_uid", hit.get("_id"))),
                score=float(hit.get("_score", 0.0)),
                rank=0,
                title=str(source.get("title", "")),
                brand=str(source.get("brand", "")),
                bullets=str(source.get("bullets", "")),
                description=str(source.get("description", "")),
            )
        )
    return sort_vector_hits(results)


def sort_vector_hits(hits: List[VectorHit]) -> List[VectorHit]:
    """Sắp xếp danh sách VectorHit theo score giảm dần rồi product_uid tăng dần và cập nhật lại rank."""
    def _key(h: VectorHit) -> Tuple[float, int, int, str]:
        uid_str = str(h.product_uid)
        is_num = 0 if uid_str.isdigit() else 1
        num_val = int(uid_str) if uid_str.isdigit() else 0
        return (-h.score, is_num, num_val, uid_str)

    sorted_hits = sorted(hits, key=_key)
    for rank, h in enumerate(sorted_hits, start=1):
        h.rank = rank
    return sorted_hits


async def search_vector(
    query: ParsedQuery,
    index_name: str = INDEX_NAME,
    k: int = RETRIEVAL_K,
) -> List[VectorHit]:
    """Truy xuất bất đồng bộ danh sách sản phẩm theo Vector k-NN."""
    return await asyncio.to_thread(_execute_vector_search, query, index_name, k)
