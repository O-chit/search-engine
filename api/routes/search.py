"""Endpoint tìm kiếm sản phẩm hỗ trợ truy xuất BM25 và phân trang."""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from api.schemas import SearchResponse, SearchResultItem, SearchTiming
from config import CANDIDATE_K, RETRIEVAL_K, SNIPPET_CHARS
from query.parser import EmptyQueryError, parse
from retrieval.bm25 import BM25RetrievalError, search_bm25

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
    t_start = time.perf_counter()

    # 1. Parse
    t_parse_start = time.perf_counter()
    try:
        parsed = parse(q)
    except EmptyQueryError as exc:
        raise HTTPException(
            status_code=400,
            detail="Chuỗi truy vấn rỗng sau khi chuẩn hóa.",
        ) from exc
    parse_ms = (time.perf_counter() - t_parse_start) * 1000

    # 2. Retrieval
    t_retrieval_start = time.perf_counter()
    try:
        hits = await search_bm25(parsed, k=RETRIEVAL_K)
    except BM25RetrievalError as exc:
        logger.error("Lỗi khi truy xuất BM25: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ tìm kiếm BM25 tạm thời không khả dụng.",
        ) from exc
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    # 3. Ranking / candidate selection
    t_rank_start = time.perf_counter()
    candidates = hits[:CANDIDATE_K]
    total_candidates = len(candidates)

    scored_candidates = []
    if total_candidates == 1:
        scored_candidates = [(candidates[0], 1.0)]
    elif total_candidates > 1:
        scores = [c.score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            scored_candidates = [(c, 1.0) for c in candidates]
        else:
            scored_candidates = [
                (c, (c.score - min_score) / (max_score - min_score)) for c in candidates
            ]
    rank_ms = (time.perf_counter() - t_rank_start) * 1000

    # 4. Phân trang
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    page_items = scored_candidates[start_idx:end_idx]

    result_items: List[SearchResultItem] = []
    for idx, (hit, rel) in enumerate(page_items):
        pos = start_idx + idx + 1
        raw_snippet = hit.bullets.strip() if hit.bullets.strip() else hit.description.strip()
        snippet = raw_snippet[:SNIPPET_CHARS]
        score = round(rel, 4)

        debug_info: Optional[Dict[str, Any]] = None
        if debug:
            debug_info = {
                "bm25_rank": hit.rank,
                "bm25_score": round(hit.score, 4),
                "rel": score,
            }

        result_items.append(
            SearchResultItem(
                product_uid=hit.product_uid,
                title=hit.title,
                brand=hit.brand,
                snippet=snippet,
                score=score,
                position=pos,
                explored=False,
                debug=debug_info,
            )
        )

    total_ms = (time.perf_counter() - t_start) * 1000
    timing = SearchTiming(
        parse=round(parse_ms, 2),
        retrieval=round(retrieval_ms, 2),
        rank=round(rank_ms, 2),
        total=round(total_ms, 2),
    )

    query_id = f"q_{uuid.uuid4().hex[:8]}"

    # Ghi log JSON một dòng mỗi request
    log_record = {
        "query_id": query_id,
        "lang": parsed.lang,
        "degraded": False,
        "timing_ms": timing.model_dump(),
    }
    log_json = json.dumps(log_record, ensure_ascii=False)
    print(log_json, flush=True)
    logger.info(log_json)

    return SearchResponse(
        query_id=query_id,
        normalized_query=parsed.normalized,
        lang=parsed.lang,
        personalized=False,
        degraded=False,
        total=total_candidates,
        page=page,
        size=size,
        results=result_items,
        timing_ms=timing,
    )
