"""Kiểm thử thuật toán gộp RRF có trọng số theo SPEC.md mục 9.3."""

from config import CANDIDATE_K
from retrieval.fusion import reciprocal_rank_fusion


def test_fusion_acceptance_a() -> None:
    """Kiểm tra RRF_K=60, w=1/1 với tài liệu A có ở cả hai nhánh và B chỉ có ở vector."""
    bm25_hits = [{"product_uid": "A", "score": 10.0, "rank": 1}]
    vec_hits = [
        {"product_uid": "B", "score": 0.9, "rank": 1},
        {"product_uid": "other", "score": 0.8, "rank": 2},
        {"product_uid": "A", "score": 0.7, "rank": 3},
    ]

    fused = reciprocal_rank_fusion(
        bm25_hits=bm25_hits,
        vec_hits=vec_hits,
        lang="en",
        rrf_k=60,
        w_bm25=1.0,
        w_vec=1.0,
    )
    fused_map = {item.product_uid: item for item in fused}

    expected_a = 1.0 / 61.0 + 1.0 / 63.0
    expected_b = 1.0 / 61.0

    assert abs(fused_map["A"].rrf - expected_a) < 1e-9
    assert abs(fused_map["B"].rrf - expected_b) < 1e-9


def test_fusion_acceptance_b() -> None:
    """Kiểm tra với lang='vi' (w_bm25=0.3, w_vec=1.0), tài liệu chỉ có ở BM25 hạng 1."""
    bm25_hits = [{"product_uid": "doc_vi", "score": 12.0, "rank": 1}]
    vec_hits = []

    fused = reciprocal_rank_fusion(
        bm25_hits=bm25_hits,
        vec_hits=vec_hits,
        lang="vi",
        rrf_k=60,
    )
    assert len(fused) == 1
    expected_rrf = 0.3 / 61.0
    assert abs(fused[0].rrf - expected_rrf) < 1e-9


def test_fusion_acceptance_c() -> None:
    """Kiểm tra hai tài liệu cùng rrf thì product_uid nhỏ hơn đứng trước và thứ tự ổn định."""
    bm25_hits_1 = [{"product_uid": "200", "score": 10.0, "rank": 1}]
    vec_hits_1 = [{"product_uid": "100", "score": 0.9, "rank": 1}]

    for _ in range(5):
        fused = reciprocal_rank_fusion(
            bm25_hits=bm25_hits_1,
            vec_hits=vec_hits_1,
            lang="en",
            rrf_k=60,
            w_bm25=1.0,
            w_vec=1.0,
        )
        assert len(fused) == 2
        assert abs(fused[0].rrf - fused[1].rrf) < 1e-9
        assert fused[0].product_uid == "100"
        assert fused[1].product_uid == "200"


def test_fusion_acceptance_d() -> None:
    """Kiểm tra kết quả cắt đúng CANDIDATE_K."""
    bm25_hits = [
        {"product_uid": f"uid_{i}", "score": float(1000 - i), "rank": i}
        for i in range(1, 150)
    ]
    vec_hits = []

    fused = reciprocal_rank_fusion(
        bm25_hits=bm25_hits,
        vec_hits=vec_hits,
        lang="en",
        candidate_k=CANDIDATE_K,
    )
    assert len(fused) == CANDIDATE_K
