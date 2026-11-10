"""Endpoint tìm kiếm sản phẩm hỗ trợ truy xuất BM25 và phân trang."""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from api.pipeline import run_pipeline
from api.schemas import SearchResponse, SearchResultItem
from config import SNIPPET_CHARS
from query.parser import EmptyQueryError
from retrieval.bm25 import BM25RetrievalError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search_endpoint(
    q: str = Query(..., description="Từ khóa tìm kiếm"),
    page: int = Query(1, ge=1, description="Trang kết quả"),
    size: int = Query(20, ge=1, le=50, description="Số lượng kết quả mỗi trang"),
    personalize: bool = Query(True, description="Bật hoặc tắt cá nhân hóa"),
    debug: bool = Query(False, description="Bật thông tin debug"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> SearchResponse:
    """Thực hiện tìm kiếm sản phẩm với BM25, chuẩn hóa điểm số và phân trang."""
    try:
        pipeline_result = await run_pipeline(q, config_name="bm25", personalize=personalize)
    except EmptyQueryError as exc:
        raise HTTPException(
            status_code=400,
            detail="Chuỗi truy vấn rỗng sau khi chuẩn hóa.",
        ) from exc
    except BM25RetrievalError as exc:
        logger.error("Lỗi khi truy xuất BM25: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ tìm kiếm BM25 tạm thời không khả dụng.",
        ) from exc

    candidates = pipeline_result.candidates
    total_candidates = len(candidates)

    start_idx = (page - 1) * size
    end_idx = start_idx + size
    page_items = candidates[start_idx:end_idx]

    result_items: List[SearchResultItem] = []
    for idx, c in enumerate(page_items):
        pos = start_idx + idx + 1
        raw_snippet = c.bullets.strip() if c.bullets.strip() else c.description.strip()
        snippet = raw_snippet[:SNIPPET_CHARS]
        score = round(c.rel, 4)

        debug_info: Optional[Dict[str, Any]] = None
        if debug:
            debug_info = {
                "bm25_rank": c.rank,
                "bm25_score": round(c.score, 4),
                "rel": score,
            }

        result_items.append(
            SearchResultItem(
                product_uid=c.product_uid,
                title=c.title,
                brand=c.brand,
                snippet=snippet,
                score=score,
                position=pos,
                explored=False,
                debug=debug_info,
            )
        )

    # Ghi log JSON một dòng mỗi request
    log_record = {
        "query_id": pipeline_result.query_id,
        "lang": pipeline_result.parsed.lang,
        "degraded": pipeline_result.degraded,
        "timing_ms": pipeline_result.timing.model_dump(),
    }
    log_json = json.dumps(log_record, ensure_ascii=False)
    print(log_json, flush=True)
    logger.info(log_json)

    return SearchResponse(
        query_id=pipeline_result.query_id,
        normalized_query=pipeline_result.parsed.normalized,
        lang=pipeline_result.parsed.lang,
        personalized=False,
        degraded=pipeline_result.degraded,
        total=total_candidates,
        page=page,
        size=size,
        results=result_items,
        timing_ms=pipeline_result.timing,
    )
