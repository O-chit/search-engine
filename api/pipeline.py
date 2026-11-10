"""Quy trình thực thi tìm kiếm sản phẩm hợp nhất các cấu hình và tính điểm."""

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
import uuid

from api.schemas import SearchTiming
from config import CANDIDATE_K, RETRIEVAL_K
from query.parser import ParsedQuery, parse
from retrieval.bm25 import BM25Hit, search_bm25


@dataclass
class CandidateItem:
    """Thông tin chi tiết của một ứng viên sản phẩm sau các bước truy xuất và xếp hạng."""

    product_uid: str
    title: str
    brand: str
    bullets: str
    description: str
    score: float
    rank: int
    rel: float


@dataclass
class PipelineResult:
    """Kết quả hoàn chỉnh của quy trình tìm kiếm kèm siêu dữ liệu và thời gian thực thi."""

    query_id: str
    parsed: ParsedQuery
    candidates: List[CandidateItem]
    timing: SearchTiming
    degraded: bool = False


async def _run_bm25_pipeline(
    parsed: ParsedQuery,
    personalize: bool = False,
) -> Tuple[List[CandidateItem], float, float]:
    """Thực thi tìm kiếm và xếp hạng ứng viên dựa trên thuật toán BM25."""
    t_retrieval_start = time.perf_counter()
    hits: List[BM25Hit] = await search_bm25(parsed, k=RETRIEVAL_K)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    t_rank_start = time.perf_counter()
    candidates = hits[:CANDIDATE_K]
    total_candidates = len(candidates)

    scored_candidates: List[Tuple[BM25Hit, float]] = []
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
                (c, (c.score - min_score) / (max_score - min_score))
                for c in candidates
            ]
    rank_ms = (time.perf_counter() - t_rank_start) * 1000

    candidate_items: List[CandidateItem] = [
        CandidateItem(
            product_uid=hit.product_uid,
            title=hit.title,
            brand=hit.brand,
            bullets=hit.bullets,
            description=hit.description,
            score=hit.score,
            rank=hit.rank,
            rel=rel,
        )
        for hit, rel in scored_candidates
    ]

    return candidate_items, retrieval_ms, rank_ms


PIPELINE_REGISTRY: Dict[
    str,
    Callable[[ParsedQuery, bool], Coroutine[Any, Any, Tuple[List[CandidateItem], float, float]]],
] = {
    "bm25": _run_bm25_pipeline,
}


def get_supported_configs() -> List[str]:
    """Trả về danh sách tên cấu hình tìm kiếm hiện có trong hệ thống."""
    return list(PIPELINE_REGISTRY.keys())


async def run_pipeline(
    query: str,
    config_name: str = "bm25",
    personalize: bool = False,
) -> PipelineResult:
    """Chạy quy trình tìm kiếm sản phẩm theo cấu hình và trả về danh sách ứng viên kèm thời gian thực thi."""
    if config_name not in PIPELINE_REGISTRY:
        supported = get_supported_configs()
        raise ValueError(
            f"Cấu hình '{config_name}' không tồn tại. Các cấu hình hợp lệ: {supported}"
        )

    t_start = time.perf_counter()

    t_parse_start = time.perf_counter()
    parsed = parse(query)
    parse_ms = (time.perf_counter() - t_parse_start) * 1000

    handler = PIPELINE_REGISTRY[config_name]
    candidates, retrieval_ms, rank_ms = await handler(parsed, personalize)

    total_ms = (time.perf_counter() - t_start) * 1000
    timing = SearchTiming(
        parse=round(parse_ms, 2),
        retrieval=round(retrieval_ms, 2),
        rank=round(rank_ms, 2),
        total=round(total_ms, 2),
    )

    query_id = f"q_{uuid.uuid4().hex[:8]}"

    return PipelineResult(
        query_id=query_id,
        parsed=parsed,
        candidates=candidates,
        timing=timing,
        degraded=False,
    )
