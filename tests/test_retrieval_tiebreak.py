"""Kiểm thử cơ chế phá hoà tất định cho truy xuất BM25 và Vector."""

from retrieval.bm25 import BM25Hit, sort_bm25_hits
from retrieval.vector import VectorHit, sort_vector_hits


def test_bm25_tiebreak_equal_scores() -> None:
    """Kiểm tra ba hit BM25 cùng điểm phải ra thứ tự product_uid tăng dần."""
    hits = [
        BM25Hit(product_uid="300", score=10.0, rank=1),
        BM25Hit(product_uid="100", score=10.0, rank=2),
        BM25Hit(product_uid="200", score=10.0, rank=3),
    ]
    sorted_hits = sort_bm25_hits(hits)
    ordered_uids = [h.product_uid for h in sorted_hits]
    assert ordered_uids == ["100", "200", "300"]
    assert [h.rank for h in sorted_hits] == [1, 2, 3]


def test_vector_tiebreak_equal_scores() -> None:
    """Kiểm tra ba hit Vector cùng điểm phải ra thứ tự product_uid tăng dần."""
    hits = [
        VectorHit(product_uid="300", score=0.85, rank=1),
        VectorHit(product_uid="100", score=0.85, rank=2),
        VectorHit(product_uid="200", score=0.85, rank=3),
    ]
    sorted_hits = sort_vector_hits(hits)
    ordered_uids = [h.product_uid for h in sorted_hits]
    assert ordered_uids == ["100", "200", "300"]
    assert [h.rank for h in sorted_hits] == [1, 2, 3]


def test_bm25_higher_score_first_regardless_of_uid() -> None:
    """Kiểm tra hit có điểm cao hơn vẫn đứng trước bất kể product_uid."""
    hits = [
        BM25Hit(product_uid="100", score=10.0, rank=1),
        BM25Hit(product_uid="900", score=20.0, rank=2),
        BM25Hit(product_uid="200", score=10.0, rank=3),
    ]
    sorted_hits = sort_bm25_hits(hits)
    ordered_uids = [h.product_uid for h in sorted_hits]
    assert ordered_uids == ["900", "100", "200"]
    assert [h.rank for h in sorted_hits] == [1, 2, 3]


def test_vector_higher_score_first_regardless_of_uid() -> None:
    """Kiểm tra hit Vector có điểm cao hơn vẫn đứng trước bất kể product_uid."""
    hits = [
        VectorHit(product_uid="100", score=0.80, rank=1),
        VectorHit(product_uid="900", score=0.90, rank=2),
        VectorHit(product_uid="200", score=0.80, rank=3),
    ]
    sorted_hits = sort_vector_hits(hits)
    ordered_uids = [h.product_uid for h in sorted_hits]
    assert ordered_uids == ["900", "100", "200"]
    assert [h.rank for h in sorted_hits] == [1, 2, 3]
