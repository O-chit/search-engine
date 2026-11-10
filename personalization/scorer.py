"""Tính điểm xếp hạng cá nhân hóa và xử lý vị trí khám phá theo SPEC.md mục 11.3–11.5."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from config import BETA_PROFILE, BETA_RECENT, EXPLORE_POSITION, WARMUP_EVENTS


@dataclass
class ScoredCandidate:
    """Đối tượng lưu trữ thông tin điểm số sau cá nhân hóa của một ứng viên sản phẩm."""

    product_uid: str
    score: float
    rel: float
    cos_profile: float
    cos_recent: float
    explored: bool = False
    original_item: Any = None


def _compute_cosine(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    """Tính độ tương đồng cosin giữa hai vector với chuẩn L2 bảo vệ chia cho 0."""
    if vec_a is None or vec_b is None:
        return 0.0
    va = np.asarray(vec_a, dtype=np.float32)
    vb = np.asarray(vec_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = float(np.dot(va, vb))
    cos_val = dot / (norm_a * norm_b)
    return float(max(-1.0, min(1.0, cos_val)))


def score_candidates(
    candidates: List[Any],
    profile_vec: Optional[np.ndarray],
    recent_vec: Optional[np.ndarray],
    event_count: int,
    personalize: bool,
    has_model_code: bool,
    normalized_query: str,
    top_rel_candidate_title: Optional[str],
    warmup_events: int = WARMUP_EVENTS,
    beta_profile: float = BETA_PROFILE,
    beta_recent: float = BETA_RECENT,
    explore_position: int = EXPLORE_POSITION,
) -> Tuple[List[ScoredCandidate], bool]:
    """Chấm điểm và sắp xếp lại ứng viên kết hợp độ liên quan và hồ sơ phiên cá nhân hóa."""
    extracted: List[Tuple[str, float, Optional[np.ndarray], Any]] = []
    for c in candidates:
        if isinstance(c, dict):
            uid = str(c["product_uid"])
            rel = float(c["rel"])
            vec = c.get("vector")
            orig = c.get("original_item", c)
            extracted.append((uid, rel, vec, orig))
        elif isinstance(c, (list, tuple)):
            uid = str(c[0])
            rel = float(c[1])
            vec = c[2] if len(c) > 2 else None
            orig = c[3] if len(c) > 3 else c
            extracted.append((uid, rel, vec, orig))
        else:
            uid = str(getattr(c, "product_uid"))
            rel = float(getattr(c, "rel"))
            vec = getattr(c, "vector", None)
            orig = getattr(c, "original_item", c)
            extracted.append((uid, rel, vec, orig))

    should_apply = (
        personalize
        and event_count > 0
        and profile_vec is not None
        and not has_model_code
        and not (top_rel_candidate_title is not None and normalized_query == top_rel_candidate_title)
    )

    if not should_apply:
        unpersonalized_list: List[ScoredCandidate] = []
        for uid, rel, vec, orig in extracted:
            cos_p = _compute_cosine(profile_vec, vec) if profile_vec is not None else 0.0
            cos_r = _compute_cosine(recent_vec, vec) if recent_vec is not None else 0.0
            unpersonalized_list.append(
                ScoredCandidate(
                    product_uid=uid,
                    score=rel,
                    rel=rel,
                    cos_profile=cos_p,
                    cos_recent=cos_r,
                    explored=False,
                    original_item=orig,
                )
            )
        unpersonalized_list.sort(key=lambda x: (-x.score, x.product_uid))
        return unpersonalized_list, False

    lambda_val = min(1.0, float(event_count) / float(warmup_events))
    scored_list: List[ScoredCandidate] = []
    for uid, rel, vec, orig in extracted:
        cos_p = _compute_cosine(profile_vec, vec) if profile_vec is not None else 0.0
        cos_r = _compute_cosine(recent_vec, vec) if recent_vec is not None else 0.0
        final_score = rel + lambda_val * (beta_profile * cos_p + beta_recent * cos_r)
        scored_list.append(
            ScoredCandidate(
                product_uid=uid,
                score=final_score,
                rel=rel,
                cos_profile=cos_p,
                cos_recent=cos_r,
                explored=False,
                original_item=orig,
            )
        )

    scored_list.sort(key=lambda x: (-x.score, x.product_uid))

    if len(scored_list) >= 10:
        top10_rel = sorted(scored_list, key=lambda x: (-x.rel, x.product_uid))[:10]
        top10_score_uids = {x.product_uid for x in scored_list[:10]}
        displaced = [x for x in top10_rel if x.product_uid not in top10_score_uids]
        if displaced:
            displaced.sort(key=lambda x: (-x.rel, x.product_uid))
            explore_target = displaced[0]
            explore_target.explored = True
            scored_list.remove(explore_target)
            insert_idx = max(0, min(explore_position - 1, len(scored_list)))
            scored_list.insert(insert_idx, explore_target)

    return scored_list, True
