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

    all_entries = sorted(os.listdir(dir_path))
    files_with_size = []
    has_checkpoint = False
    for fname in all_entries:
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            files_with_size.append((fname, size_mb))
            if fname.startswith("checkpoint"):
                has_checkpoint = True

    print(f"Thư mục: {dir_path}")
    print(f"  Shape: {embs.shape}")
    print(f"  Dtype: {embs.dtype}")
    print(f"  Chuẩn L2 (min / max / mean): {norm_min:.6f} / {norm_max:.6f} / {norm_mean:.6f}")
    print(f"  Số mục trong uid_index.json: {n_uids}")
    print(f"  uid_index ánh xạ đúng 0..{n_uids - 1}: {is_valid_mapping}")
    print(f"  Có dòng NaN: {has_nan}")
    print(f"  Có dòng toàn 0: {has_zeros}")
    print(f"  Danh sách file:")
    for fname, size_mb in files_with_size:
        print(f"    - {fname}: {size_mb:.2f} MB")
    print(f"  còn file checkpoint: {has_checkpoint}")

    return embs, uid_index


def check_parquet(parquet_path: str, uid_index: Dict[str, int]) -> None:
    """Kiểm tra tính khớp giữa product_uid trong parquet và uid_index."""
    import pandas as pd

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Không tìm thấy file parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    parquet_uids = df["product_uid"].astype(str).tolist()
    n_parquet = len(parquet_uids)
    n_index = len(uid_index)

    set_parquet = set(parquet_uids)
    set_index = set(uid_index.keys())
    sets_match = set_parquet == set_index

    is_in_order = (n_parquet == n_index) and all(
        uid_index.get(uid) == i for i, uid in enumerate(parquet_uids)
    )

    print(f"Kiểm tra với Parquet: {parquet_path}")
    print(f"  Số product_uid trong parquet: {n_parquet}")
    print(f"  Số mục trong uid_index: {n_index}")
    print(f"  Tập product_uid trùng khớp hoàn toàn: {sets_match}")
    print(f"  Chỉ số dòng theo đúng thứ tự dòng của parquet: {is_in_order}")


def compare_embeddings(
    embs1: np.ndarray,
    uid_index1: Dict[str, int],
    embs2: np.ndarray,
    uid_index2: Dict[str, int],
) -> None:
    """So sánh sự tương đồng giữa hai tập embedding và uid_index."""
    uids_identical = uid_index1 == uid_index2
    print("So sánh hai thư mục:")
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
    parser.add_argument(
        "--check-parquet",
        default=None,
        help="Đường dẫn file parquet để kiểm tra tính khớp của product_uid và thứ tự",
    )
    args = parser.parse_args()

    embs1, uid_index1 = inspect_directory(args.dir)
    if args.check_parquet:
        check_parquet(args.check_parquet, uid_index1)
    if args.compare:
        embs2, uid_index2 = inspect_directory(args.compare)
        compare_embeddings(embs1, uid_index1, embs2, uid_index2)


if __name__ == "__main__":
    main()
