"""Các hàm đo lường chất lượng tìm kiếm thuần túy (Recall@k, MRR, NDCG@10)."""

import math
from typing import Dict, List, Optional


def label_from_relevance(relevance: float) -> int:
    """Quy đổi điểm relevance (1.0-3.0) thành nhãn nguyên rời rạc (0-4)."""
    return int(round((relevance - 1.0) * 2.0))


def recall_at_k(
    retrieved_uids: List[str],
    ground_truth: Dict[str, float],
    k: int = 100,
    threshold: float = 2.5,
) -> Optional[float]:
    """Tính tỷ lệ sản phẩm có relevance >= threshold nằm trong k ứng viên hàng đầu."""
    relevant_uids = {uid for uid, rel in ground_truth.items() if rel >= threshold}
    if not relevant_uids:
        return None
    top_k_set = set(retrieved_uids[:k])
    hits = len(relevant_uids & top_k_set)
    return hits / len(relevant_uids)


def mrr(
    retrieved_uids: List[str],
    ground_truth: Dict[str, float],
    threshold: float = 2.5,
) -> Optional[float]:
    """Tính điểm Mean Reciprocal Rank (MRR) dựa trên vị trí sản phẩm phù hợp đầu tiên."""
    relevant_uids = {uid for uid, rel in ground_truth.items() if rel >= threshold}
    if not relevant_uids:
        return None
    for pos, uid in enumerate(retrieved_uids, start=1):
        if uid in relevant_uids:
            return 1.0 / pos
    return 0.0


def ndcg_at_10_truncated(
    retrieved_uids: List[str],
    ground_truth: Dict[str, float],
    k: int = 10,
    candidate_k: int = 100,
) -> Optional[float]:
    """Tính chỉ số NDCG@10 rút gọn trên tập sản phẩm có nhãn của truy vấn."""
    if not ground_truth:
        return None

    labels = {uid: label_from_relevance(rel) for uid, rel in ground_truth.items()}

    seen = set()
    in_candidates: List[str] = []
    for uid in retrieved_uids[:candidate_k]:
        if uid in labels and uid not in seen:
            in_candidates.append(uid)
            seen.add(uid)

    not_in_candidates = sorted([uid for uid in labels if uid not in seen])
    ranked_labeled = in_candidates + not_in_candidates

    # Tính DCG
    dcg = 0.0
    for i, uid in enumerate(ranked_labeled[:k], start=1):
        gain = (2.0 ** labels[uid]) - 1.0
        discount = math.log2(i + 1.0)
        dcg += gain / discount

    # Tính IDCG
    ideal_labels = sorted(labels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, lbl in enumerate(ideal_labels, start=1):
        gain = (2.0 ** lbl) - 1.0
        discount = math.log2(i + 1.0)
        idcg += gain / discount

    if idcg == 0.0:
        return None

    return dcg / idcg
