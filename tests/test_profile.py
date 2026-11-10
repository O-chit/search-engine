"""Kiểm thử đơn vị các hàm thuần trong mô-đun quản lý hồ sơ sở thích cá nhân hóa."""

import numpy as np
import pytest

from personalization.profile import (
    compute_recent_vec,
    is_valid_click,
    update_profile_vec,
    update_recent_uids,
)


def test_profile_update_empty() -> None:
    """Kiểm tra cập nhật khi hồ sơ chưa tồn tại trả về vector chuẩn hóa của e."""
    e = np.array([1.0, 0.0], dtype=np.float32)
    res = update_profile_vec(None, e, eta=0.2)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-6)


def test_profile_update_existing() -> None:
    """Kiểm tra cập nhật trung bình trượt với ETA=0.2 và chuẩn hóa độ dài 1."""
    profile = np.array([1.0, 0.0], dtype=np.float32)
    e = np.array([0.0, 1.0], dtype=np.float32)
    res = update_profile_vec(profile, e, eta=0.2)
    expected = np.array([0.9701425, 0.2425356], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-6)


def test_update_recent_uids() -> None:
    """Kiểm tra đẩy danh sách sản phẩm gần đây với giới hạn tối đa và loại trùng."""
    uids = []
    for i in range(1, 8):
        uids = update_recent_uids(uids, f"u{i}", max_items=5)
    assert uids == ["u7", "u6", "u5", "u4", "u3"]

    uids = update_recent_uids(uids, "u5", max_items=5)
    assert uids == ["u5", "u7", "u6", "u4", "u3"]


def test_is_valid_click() -> None:
    """Kiểm tra tính hợp lệ của lượt click theo thời gian và mã sản phẩm."""
    t0 = 1000.0
    # Chưa có click trước -> hợp lệ
    assert is_valid_click(None, None, "a", t0) is True

    # Click trước 'a' lúc t, click 'a' lúc t+1.5s -> không hợp lệ
    assert is_valid_click("a", t0, "a", t0 + 1.5, dedupe_seconds=2.0) is False

    # Click trước 'a' lúc t, click 'a' lúc t+2.5s -> hợp lệ
    assert is_valid_click("a", t0, "a", t0 + 2.5, dedupe_seconds=2.0) is True

    # Click trước 'a' lúc t, click 'b' lúc t+0.5s -> hợp lệ
    assert is_valid_click("a", t0, "b", t0 + 0.5, dedupe_seconds=2.0) is True


def test_compute_recent_vec() -> None:
    """Kiểm tra tính trung bình vector gần đây và chuẩn hóa L2 về độ dài 1."""
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    res = compute_recent_vec([v1, v2])
    assert res is not None
    expected = np.array([0.7071068, 0.7071068], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-6)

    # Danh sách rỗng -> None
    assert compute_recent_vec([]) is None


def test_warmup_queries_do_not_contain_test_queries() -> None:
    """Kiểm tra 50 truy vấn làm nóng ngẫu nhiên từ tập train không chứa 3 truy vấn kiểm tra."""
    import json
    import os
    import random
    from config import SPLIT_PATH, WARMUP_QUERY_COUNT, WARMUP_SEED

    if not os.path.exists(SPLIT_PATH):
        return
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_queries = data.get("train", [])
    rng = random.Random(WARMUP_SEED)
    warmup_queries = set(rng.sample(train_queries, min(WARMUP_QUERY_COUNT, len(train_queries))))

    test_queries = [
        "copper pipe fittings",
        "solar path lights",
        "smoke detector battery",
    ]
    for q in test_queries:
        assert q not in warmup_queries, f"Truy vấn '{q}' không được nằm trong warmup_queries"
