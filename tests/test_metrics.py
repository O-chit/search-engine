"""Kiểm thử đơn vị cho các hàm tính chỉ số đánh giá (Recall@k, MRR, NDCG@10)."""

import pytest

from eval.metrics import (
    label_from_relevance,
    mrr,
    ndcg_at_10_truncated,
    recall_at_k,
)


def test_label_from_relevance() -> None:
    """Kiểm tra quy đổi điểm relevance thành nhãn nguyên rời rạc."""
    assert label_from_relevance(1.0) == 0
    assert label_from_relevance(2.5) == 3
    assert label_from_relevance(3.0) == 4


def test_spec_example_metrics() -> None:
    """Kiểm tra ví dụ cụ thể trong đặc tả kỹ thuật với sai số tối đa 1e-4."""
    # Hệ thống trả ['p1', 'p2', 'p3']; nhãn relevance p1=1.0, p2=3.0, p4=2.5 (p4 không có trong ứng viên)
    retrieved = ["p1", "p2", "p3"]
    ground_truth = {"p1": 1.0, "p2": 3.0, "p4": 2.5}

    # Kỳ vọng:
    # label_from_relevance(1.0)=0, (3.0)=4, (2.5)=3
    assert label_from_relevance(1.0) == 0
    assert label_from_relevance(3.0) == 4
    assert label_from_relevance(2.5) == 3

    # recall_at_k=0.5
    rec = recall_at_k(retrieved, ground_truth, k=100)
    assert rec is not None
    assert pytest.approx(rec, abs=1e-4) == 0.5

    # mrr=0.5
    m = mrr(retrieved, ground_truth)
    assert m is not None
    assert pytest.approx(m, abs=1e-4) == 0.5

    # ndcg_at_10_truncated=0.6677 (DCG 12.9639, IDCG 19.4165)
    ndcg = ndcg_at_10_truncated(retrieved, ground_truth, k=10, candidate_k=100)
    assert ndcg is not None
    assert pytest.approx(ndcg, abs=1e-4) == 0.6677


def test_ideal_ndcg() -> None:
    """Kiểm tra hệ thống xếp đúng thứ tự lý tưởng phải đạt NDCG bằng 1.0."""
    retrieved_ideal = ["p2", "p4", "p1"]
    ground_truth = {"p1": 1.0, "p2": 3.0, "p4": 2.5}
    ndcg = ndcg_at_10_truncated(retrieved_ideal, ground_truth, k=10, candidate_k=100)
    assert ndcg is not None
    assert pytest.approx(ndcg, abs=1e-4) == 1.0


def test_no_relevant_products_returns_none() -> None:
    """Kiểm tra truy vấn không có sản phẩm nào >= 2.5 thì recall và mrr trả về None."""
    retrieved = ["p1", "p2", "p3"]
    ground_truth_low = {"p1": 1.0, "p2": 2.0, "p3": 2.2}
    assert recall_at_k(retrieved, ground_truth_low, k=100) is None
    assert mrr(retrieved, ground_truth_low) is None


def test_idcg_zero_returns_none() -> None:
    """Kiểm tra truy vấn có IDCG bằng 0 (mọi nhãn bằng 0) thì NDCG trả về None."""
    retrieved = ["p1", "p2"]
    ground_truth_zero = {"p1": 1.0, "p2": 1.0}
    assert ndcg_at_10_truncated(retrieved, ground_truth_zero, k=10) is None
