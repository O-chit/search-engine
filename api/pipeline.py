"""Quy trình thực thi tìm kiếm sản phẩm hợp nhất các cấu hình và tính điểm."""

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
import uuid

from api.schemas import SearchTiming
from config import BRANCH_TIMEOUT_MS, CANDIDATE_K, RETRIEVAL_K
from query.parser import ParsedQuery, parse
from retrieval.bm25 import BM25Hit, search_bm25
from retrieval.fusion import FusedCandidate, reciprocal_rank_fusion
from retrieval.vector import VectorHit, search_vector

logger = logging.getLogger(__name__)


class HybridSearchError(Exception):
    """Lỗi xảy ra khi cả hai nhánh tìm kiếm trong hybrid đều thất bại."""

    pass


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
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    vec_rank: Optional[int] = None
    vec_score: Optional[float] = None
    rrf: Optional[float] = None


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
) -> Tuple[List[CandidateItem], float, float, bool]:
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
            bm25_rank=hit.rank,
            bm25_score=hit.score,
            vec_rank=None,
            vec_score=None,
            rrf=None,
        )
        for hit, rel in scored_candidates
    ]

    return candidate_items, retrieval_ms, rank_ms, False


async def _run_vector_pipeline(
    parsed: ParsedQuery,
    personalize: bool = False,
) -> Tuple[List[CandidateItem], float, float, bool]:
    """Thực thi tìm kiếm và xếp hạng ứng viên dựa trên thuật toán Vector."""
    t_retrieval_start = time.perf_counter()
    hits: List[VectorHit] = await search_vector(parsed, k=RETRIEVAL_K)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    t_rank_start = time.perf_counter()
    candidates = hits[:CANDIDATE_K]
    total_candidates = len(candidates)

    scored_candidates: List[Tuple[VectorHit, float]] = []
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
            bm25_rank=None,
            bm25_score=None,
            vec_rank=hit.rank,
            vec_score=hit.score,
            rrf=None,
        )
        for hit, rel in scored_candidates
    ]

    return candidate_items, retrieval_ms, rank_ms, False


async def _run_hybrid_pipeline(
    parsed: ParsedQuery,
    personalize: bool = False,
) -> Tuple[List[CandidateItem], float, float, bool]:
    """Thực thi tìm kiếm song song BM25 và Vector, gộp bằng RRF và xếp hạng."""
    t_retrieval_start = time.perf_counter()
    timeout_sec = BRANCH_TIMEOUT_MS / 1000.0

    async def _fetch_bm25() -> List[BM25Hit]:
        return await asyncio.wait_for(search_bm25(parsed, k=RETRIEVAL_K), timeout=timeout_sec)

    async def _fetch_vector() -> List[VectorHit]:
        return await asyncio.wait_for(search_vector(parsed, k=RETRIEVAL_K), timeout=timeout_sec)

    results = await asyncio.gather(_fetch_bm25(), _fetch_vector(), return_exceptions=True)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    bm25_res, vec_res = results[0], results[1]

    bm25_hits: List[BM25Hit] = []
    bm25_failed = False
    if isinstance(bm25_res, Exception):
        logger.warning("Nhánh BM25 lỗi hoặc quá hạn: %s", bm25_res, exc_info=bm25_res)
        bm25_failed = True
    else:
        bm25_hits = bm25_res

    vec_hits: List[VectorHit] = []
    vec_failed = False
    if isinstance(vec_res, Exception):
        logger.warning("Nhánh Vector lỗi hoặc quá hạn: %s", vec_res, exc_info=vec_res)
        vec_failed = True
    else:
        vec_hits = vec_res

    if bm25_failed and vec_failed:
        logger.error("Cả hai nhánh tìm kiếm đều thất bại: BM25=%s, Vector=%s", bm25_res, vec_res)
        raise HybridSearchError(f"Cả hai nhánh tìm kiếm đều thất bại: BM25: {bm25_res}, Vector: {vec_res}")

    degraded = bm25_failed or vec_failed

    t_rank_start = time.perf_counter()
    fused: List[FusedCandidate] = reciprocal_rank_fusion(
        bm25_hits=bm25_hits,
        vec_hits=vec_hits,
        lang=parsed.lang,
        candidate_k=CANDIDATE_K,
    )

    total_candidates = len(fused)
    scored_fused: List[Tuple[FusedCandidate, float]] = []
    if total_candidates == 1:
        scored_fused = [(fused[0], 1.0)]
    elif total_candidates > 1:
        rrfs = [c.rrf for c in fused]
        min_rrf = min(rrfs)
        max_rrf = max(rrfs)
        if max_rrf == min_rrf:
            scored_fused = [(c, 1.0) for c in fused]
        else:
            scored_fused = [
                (c, (c.rrf - min_rrf) / (max_rrf - min_rrf))
                for c in fused
            ]
    rank_ms = (time.perf_counter() - t_rank_start) * 1000

    candidate_items: List[CandidateItem] = [
        CandidateItem(
            product_uid=c.product_uid,
            title=c.title,
            brand=c.brand,
            bullets=c.bullets,
            description=c.description,
            score=c.rrf,
            rank=rank,
            rel=rel,
            bm25_rank=c.bm25_rank,
            bm25_score=c.bm25_score,
            vec_rank=c.vec_rank,
            vec_score=c.vec_score,
            rrf=c.rrf,
        )
        for rank, (c, rel) in enumerate(scored_fused, start=1)
    ]

    return candidate_items, retrieval_ms, rank_ms, degraded


PIPELINE_REGISTRY: Dict[
    str,
    Callable[[ParsedQuery, bool], Coroutine[Any, Any, Tuple[List[CandidateItem], float, float, bool]]],
] = {
    "bm25": _run_bm25_pipeline,
    "vector": _run_vector_pipeline,
    "hybrid": _run_hybrid_pipeline,
}


def get_supported_configs() -> List[str]:
    """Trả về danh sách tên cấu hình tìm kiếm hiện có trong hệ thống."""
    return list(PIPELINE_REGISTRY.keys())


async def run_pipeline(
    query: str,
    config_name: str = "hybrid",
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
    with_vector = config_name in ("vector", "hybrid")
    parsed = parse(query, with_vector=with_vector)
    parse_ms = (time.perf_counter() - t_parse_start) * 1000

    handler = PIPELINE_REGISTRY[config_name]
    candidates, retrieval_ms, rank_ms, degraded = await handler(parsed, personalize)

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
        degraded=degraded,
    )
