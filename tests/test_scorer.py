"""Kiểm thử đơn vị các hàm thuần trong mô-đun chấm điểm cá nhân hóa và vị trí khám phá."""

import numpy as np
import pytest

from personalization.scorer import score_candidates


def test_scorer_basic_cases() -> None:
    """Kiểm tra các trường hợp cơ bản chấm điểm cá nhân hóa theo số sự kiện và cờ bảo vệ."""
    cand_a = ("A", 1.0, np.array([1.0, 0.0], dtype=np.float32))
    cand_b = ("B", 0.9, np.array([0.0, 1.0], dtype=np.float32))
    candidates = [cand_a, cand_b]

    profile_vec = np.array([0.0, 1.0], dtype=np.float32)
    recent_vec = np.array([0.0, 1.0], dtype=np.float32)

    # (a) event_count 0 -> score == rel cho mọi ứng viên, thứ tự A,B, applied False
    res_a, applied_a = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=0,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_a is False
    assert [x.product_uid for x in res_a] == ["A", "B"]
    assert res_a[0].score == 1.0
    assert res_a[1].score == 0.9

    # (b) event_count 1 -> score B = 0.98, A = 1.0, thứ tự A,B, applied True
    res_b, applied_b = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=1,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_b is True
    assert [x.product_uid for x in res_b] == ["A", "B"]
    assert pytest.approx(res_b[0].score, abs=1e-5) == 1.0
    assert pytest.approx(res_b[1].score, abs=1e-5) == 0.98

    # (c) event_count 2 -> score B = 1.06, thứ tự B,A
    res_c, applied_c = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=2,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_c is True
    assert [x.product_uid for x in res_c] == ["B", "A"]
    assert pytest.approx(res_c[0].score, abs=1e-5) == 1.06
    assert pytest.approx(res_c[1].score, abs=1e-5) == 1.00

    # (d) event_count 10 -> score B = 1.3
    res_d, applied_d = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_d is True
    assert pytest.approx(res_d[0].score, abs=1e-5) == 1.3
    assert res_d[0].product_uid == "B"

    # (e) event_count 10 nhưng has_model_code=True -> score == rel, applied False
    res_e, applied_e = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=True,
        has_model_code=True,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_e is False
    assert res_e[0].score == res_e[0].rel
    assert res_e[1].score == res_e[1].rel

    # (f) event_count 10 nhưng normalized truy vấn trùng title chuẩn hoá của ứng viên rel cao nhất -> score == rel, applied False
    res_f, applied_f = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=True,
        has_model_code=False,
        normalized_query="top title",
        top_rel_candidate_title="top title",
    )
    assert applied_f is False
    assert res_f[0].score == res_f[0].rel
    assert res_f[1].score == res_f[1].rel

    # (g) personalize=False -> score == rel, applied False
    res_g, applied_g = score_candidates(
        candidates,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=False,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_g is False
    assert res_g[0].score == res_g[0].rel
    assert res_g[1].score == res_g[1].rel


def test_scorer_exploration_position() -> None:
    """Kiểm tra xử lý vị trí khám phá 11.5 khi ứng viên top 10 rel bị đẩy ra ngoài top 10 score."""
    vec_match = np.array([0.0, 1.0], dtype=np.float32)
    vec_non_match = np.array([1.0, 0.0], dtype=np.float32)
    profile_vec = np.array([0.0, 1.0], dtype=np.float32)
    recent_vec = np.array([0.0, 1.0], dtype=np.float32)

    # 12 ứng viên: C01..C09 khớp profile (rel 1.00..0.92)
    # C10 không khớp (rel 0.90, top 10 rel nhưng score không tăng)
    # C11 khớp profile (rel 0.80, score tăng 0.40 -> 1.20 nhảy vào top 10)
    # C12 không khớp (rel 0.50)
    candidates_12 = []
    for i in range(1, 10):
        candidates_12.append((f"C{i:02d}", 1.0 - 0.01 * (i - 1), vec_match))
    candidates_12.append(("C10", 0.90, vec_non_match))
    candidates_12.append(("C11", 0.80, vec_match))
    candidates_12.append(("C12", 0.50, vec_non_match))

    res_12, applied_12 = score_candidates(
        candidates_12,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_12 is True
    # (h) Ứng viên C10 nằm ở vị trí 8 (chỉ số 7), explored True, mọi ứng viên khác explored False
    assert res_12[7].product_uid == "C10"
    assert res_12[7].explored is True
    for idx, cand in enumerate(res_12):
        if idx != 7:
            assert cand.explored is False

    # (i) chỉ 9 ứng viên -> không ứng viên nào explored
    candidates_9 = candidates_12[:9]
    res_9, applied_9 = score_candidates(
        candidates_9,
        profile_vec=profile_vec,
        recent_vec=recent_vec,
        event_count=10,
        personalize=True,
        has_model_code=False,
        normalized_query="test query",
        top_rel_candidate_title="other title",
    )
    assert applied_9 is True
    for cand in res_9:
        assert cand.explored is False
