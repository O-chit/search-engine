"""Kiểm tra và so sánh các file vector embedding và uid_index."""

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def inspect_directory(dir_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Đọc và kiểm tra các thông số kỹ thuật của embedding và uid_index trong thư mục."""
    embs_path = os.path.join(dir_path, "embeddings.npy")
    uids_path = os.path.join(dir_path, "uid_index.json")

    if not os.path.exists(embs_path):
        raise FileNotFoundError(f"Không tìm thấy file {embs_path}")
    if not os.path.exists(uids_path):
        raise FileNotFoundError(f"Không tìm thấy file {uids_path}")

    embs = np.load(embs_path)
    with open(uids_path, "r", encoding="utf-8") as f:
        uid_index = json.load(f)

    n_rows = len(embs)
    norms = np.linalg.norm(embs, axis=1)
    norm_min = float(np.min(norms)) if n_rows > 0 else 0.0
    norm_max = float(np.max(norms)) if n_rows > 0 else 0.0
    norm_mean = float(np.mean(norms)) if n_rows > 0 else 0.0

    n_uids = len(uid_index)
    sorted_vals = sorted(uid_index.values())
    is_valid_mapping = sorted_vals == list(range(n_uids))

    has_nan = bool(np.isnan(embs).any())
    has_zeros = bool(np.all(embs == 0, axis=1).any()) if n_rows > 0 else False

    print(f"Thư mục: {dir_path}")
    print(f"  Shape: {embs.shape}")
    print(f"  Dtype: {embs.dtype}")
    print(f"  Chuẩn L2 (min / max / mean): {norm_min:.6f} / {norm_max:.6f} / {norm_mean:.6f}")
    print(f"  Số mục trong uid_index.json: {n_uids}")
    print(f"  uid_index ánh xạ đúng 0..{n_uids - 1}: {is_valid_mapping}")
    print(f"  Có dòng NaN: {has_nan}")
    print(f"  Có dòng toàn 0: {has_zeros}")

    return embs, uid_index


def compare_embeddings(
    embs1: np.ndarray,
    uid_index1: Dict[str, int],
    embs2: np.ndarray,
    uid_index2: Dict[str, int],
) -> None:
    """So sánh sự tương đồng giữa hai tập embedding và uid_index."""
    uids_identical = uid_index1 == uid_index2
    print(f"So sánh hai thư mục:")
    print(f"  Hai uid_index giống hệt: {uids_identical}")

    if embs1.shape != embs2.shape:
        print(f"  Shape khác nhau: {embs1.shape} != {embs2.shape}")
    else:
        max_diff = float(np.max(np.abs(embs1 - embs2)))
        print(f"  Max |a-b|: {max_diff:.6e}")


def main() -> None:
    """Điểm vào chính chạy inspect embeddings từ dòng lệnh."""
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng vector embedding.")
    parser.add_argument("--dir", required=True, help="Thư mục chứa embeddings.npy và uid_index.json")
    parser.add_argument(
        "--compare",
        default=None,
        help="Thư mục thứ hai để so sánh sai khác vector",
    )
    args = parser.parse_args()

    embs1, uid_index1 = inspect_directory(args.dir)
    if args.compare:
        embs2, uid_index2 = inspect_directory(args.compare)
        compare_embeddings(embs1, uid_index1, embs2, uid_index2)


if __name__ == "__main__":
    main()
