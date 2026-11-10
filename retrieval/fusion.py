"""Mô-đun gộp kết quả tìm kiếm BM25 và Vector bằng thuật toán RRF có trọng số."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import (
    CANDIDATE_K,
    RRF_K,
    W_BM25_EN,
    W_BM25_VI,
    W_VEC_EN,
    W_VEC_VI,
)


@dataclass
class FusedCandidate:
    """Ứng viên sau khi kết hợp kết quả từ BM25 và Vector bằng RRF."""

    product_uid: str
    rrf: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vec_score: Optional[float] = None
    vec_rank: Optional[int] = None
    title: str = ""
    brand: str = ""
    bullets: str = ""
    description: str = ""


def _extract_hit_data(hit: Any, default_rank: int) -> Tuple[str, float, int, str, str, str, str]:
    """Trích xuất thông tin product_uid, score, rank và metadata từ hit."""
    if isinstance(hit, dict):
        uid = str(hit.get("product_uid", hit.get("_id", "")))
        score = float(hit.get("score", hit.get("_score", 0.0)))
        rank = int(hit.get("rank", default_rank))
        title = str(hit.get("title", ""))
        brand = str(hit.get("brand", ""))
        bullets = str(hit.get("bullets", ""))
        description = str(hit.get("description", ""))
    else:
        uid = str(getattr(hit, "product_uid", getattr(hit, "_id", "")))
        score = float(getattr(hit, "score", getattr(hit, "_score", 0.0)))
        rank = int(getattr(hit, "rank", default_rank))
        title = str(getattr(hit, "title", ""))
        brand = str(getattr(hit, "brand", ""))
        bullets = str(getattr(hit, "bullets", ""))
        description = str(getattr(hit, "description", ""))
    return uid, score, rank, title, brand, bullets, description


def reciprocal_rank_fusion(
    bm25_hits: List[Any],
    vec_hits: List[Any],
    lang: str = "en",
    rrf_k: int = RRF_K,
    candidate_k: int = CANDIDATE_K,
    w_bm25: Optional[float] = None,
    w_vec: Optional[float] = None,
) -> List[FusedCandidate]:
    """Gộp danh sách ứng viên từ nhánh BM25 và Vector bằng thuật toán RRF có trọng số."""
    if w_bm25 is None or w_vec is None:
        if lang == "vi":
            w_bm25 = W_BM25_VI if w_bm25 is None else w_bm25
            w_vec = W_VEC_VI if w_vec is None else w_vec
        else:
            w_bm25 = W_BM25_EN if w_bm25 is None else w_bm25
            w_vec = W_VEC_EN if w_vec is None else w_vec

    candidates_map: Dict[str, Dict[str, Any]] = {}

    for idx, hit in enumerate(bm25_hits, start=1):
        uid, score, rank, title, brand, bullets, desc = _extract_hit_data(hit, default_rank=idx)
        if not uid:
            continue
        if uid not in candidates_map:
            candidates_map[uid] = {
                "product_uid": uid,
                "title": title,
                "brand": brand,
                "bullets": bullets,
                "description": desc,
                "bm25_score": score,
                "bm25_rank": rank,
                "vec_score": None,
                "vec_rank": None,
            }
        else:
            candidates_map[uid]["bm25_score"] = score
            candidates_map[uid]["bm25_rank"] = rank
            if not candidates_map[uid]["title"]:
                candidates_map[uid]["title"] = title
            if not candidates_map[uid]["brand"]:
                candidates_map[uid]["brand"] = brand
            if not candidates_map[uid]["bullets"]:
                candidates_map[uid]["bullets"] = bullets
            if not candidates_map[uid]["description"]:
                candidates_map[uid]["description"] = desc

    for idx, hit in enumerate(vec_hits, start=1):
        uid, score, rank, title, brand, bullets, desc = _extract_hit_data(hit, default_rank=idx)
        if not uid:
            continue
        if uid not in candidates_map:
            candidates_map[uid] = {
                "product_uid": uid,
                "title": title,
                "brand": brand,
                "bullets": bullets,
                "description": desc,
                "bm25_score": None,
                "bm25_rank": None,
                "vec_score": score,
                "vec_rank": rank,
            }
        else:
            candidates_map[uid]["vec_score"] = score
            candidates_map[uid]["vec_rank"] = rank
            if not candidates_map[uid]["title"]:
                candidates_map[uid]["title"] = title
            if not candidates_map[uid]["brand"]:
                candidates_map[uid]["brand"] = brand
            if not candidates_map[uid]["bullets"]:
                candidates_map[uid]["bullets"] = bullets
            if not candidates_map[uid]["description"]:
                candidates_map[uid]["description"] = desc

    fused_list: List[FusedCandidate] = []
    for uid, data in candidates_map.items():
        score_bm25_part = 0.0
        if data["bm25_rank"] is not None:
            score_bm25_part = w_bm25 / (rrf_k + data["bm25_rank"])

        score_vec_part = 0.0
        if data["vec_rank"] is not None:
            score_vec_part = w_vec / (rrf_k + data["vec_rank"])

        total_rrf = score_bm25_part + score_vec_part

        fused_list.append(
            FusedCandidate(
                product_uid=uid,
                rrf=total_rrf,
                bm25_score=data["bm25_score"],
                bm25_rank=data["bm25_rank"],
                vec_score=data["vec_score"],
                vec_rank=data["vec_rank"],
                title=data["title"],
                brand=data["brand"],
                bullets=data["bullets"],
                description=data["description"],
            )
        )

    def _sort_key(c: FusedCandidate) -> Tuple[float, int, int, str]:
        uid_str = str(c.product_uid)
        is_num = 0 if uid_str.isdigit() else 1
        num_val = int(uid_str) if uid_str.isdigit() else 0
        return (-c.rrf, is_num, num_val, uid_str)

    fused_list.sort(key=_sort_key)
    return fused_list[:candidate_k]
