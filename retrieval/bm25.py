"""Truy xuất tài liệu bằng thuật toán BM25 qua OpenSearch."""

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple

from config import BRANCH_TIMEOUT_MS, INDEX_NAME, RETRIEVAL_K
from query.parser import ParsedQuery
from retrieval.client import get_opensearch_client

logger = logging.getLogger(__name__)


class BM25RetrievalError(Exception):
    """Lỗi xảy ra trong quá trình truy xuất BM25 từ OpenSearch."""
    pass


@dataclass
class BM25Hit:
    """Cấu trúc dữ liệu kết quả trả về từ nhánh BM25."""

    product_uid: str
    score: float
    rank: int
    title: str = ""
    brand: str = ""
    bullets: str = ""
    description: str = ""


def _execute_bm25_search(
    query: ParsedQuery,
    index_name: str,
    k: int,
) -> List[BM25Hit]:
    """Thực thi truy vấn OpenSearch BM25 đồng bộ."""
    client = get_opensearch_client()
    min_should_match = "1" if query.lang == "vi" else "75%"
    query_body: Dict[str, Any] = {
        "size": k,
        "timeout": f"{BRANCH_TIMEOUT_MS}ms",
        "_source": ["product_uid", "title", "brand", "bullets", "description"],
        "query": {
            "multi_match": {
                "query": query.normalized,
                "type": "best_fields",
                "tie_breaker": 0.3,
                "fields": [
                    "title^3.0",
                    "brand^2.0",
                    "bullets^1.5",
                    "attributes_text^1.0",
                    "description^1.0",
                ],
                "minimum_should_match": min_should_match,
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
        logger.error("Lỗi khi gọi tìm kiếm BM25 trên OpenSearch: %s", exc, exc_info=True)
        raise BM25RetrievalError(f"OpenSearch search failed: {exc}") from exc

    hits = response.get("hits", {}).get("hits", [])
    results: List[BM25Hit] = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append(
            BM25Hit(
                product_uid=str(source.get("product_uid", hit.get("_id"))),
                score=float(hit.get("_score", 0.0)),
                rank=0,
                title=str(source.get("title", "")),
                brand=str(source.get("brand", "")),
                bullets=str(source.get("bullets", "")),
                description=str(source.get("description", "")),
            )
        )
    return sort_bm25_hits(results)


def sort_bm25_hits(hits: List[BM25Hit]) -> List[BM25Hit]:
    """Sắp xếp danh sách BM25Hit theo score giảm dần rồi product_uid tăng dần và cập nhật lại rank."""
    def _key(h: BM25Hit) -> Tuple[float, int, int, str]:
        uid_str = str(h.product_uid)
        is_num = 0 if uid_str.isdigit() else 1
        num_val = int(uid_str) if uid_str.isdigit() else 0
        return (-h.score, is_num, num_val, uid_str)

    sorted_hits = sorted(hits, key=_key)
    for rank, h in enumerate(sorted_hits, start=1):
        h.rank = rank
    return sorted_hits


async def search_bm25(
    query: ParsedQuery,
    index_name: str = INDEX_NAME,
    k: int = RETRIEVAL_K,
) -> List[BM25Hit]:
    """Truy xuất bất đồng bộ danh sách sản phẩm theo BM25."""
    return await asyncio.to_thread(_execute_bm25_search, query, index_name, k)
