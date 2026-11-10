"""Chia tập truy vấn train/test ổn định từ train.csv phục vụ đánh giá."""

import hashlib
import json
import os
import random
from typing import Dict, List

from config import (
    SPLIT_PATH,
    SPLIT_SEED,
    SPLIT_TEST_RATIO,
    TRAIN_CSV_PATH,
)
from ingestion.load_homedepot import read_csv_robust


def compute_file_md5(file_path: str) -> str:
    """Tính mã băm MD5 của một tệp tin."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_search_terms(
    csv_path: str = TRAIN_CSV_PATH,
    test_ratio: float = SPLIT_TEST_RATIO,
    seed: int = SPLIT_SEED,
) -> Dict[str, List[str]]:
    """Phân chia các truy vấn duy nhất từ train.csv thành tập train và test có sắp xếp ổn định."""
    df = read_csv_robust(csv_path, dtype={"search_term": str})
    unique_terms = sorted(df["search_term"].dropna().astype(str).unique().tolist())

    rng = random.Random(seed)
    shuffled_terms = list(unique_terms)
    rng.shuffle(shuffled_terms)

    n_train = int(len(shuffled_terms) * (1.0 - test_ratio))
    train_terms = sorted(shuffled_terms[:n_train])
    test_terms = sorted(shuffled_terms[n_train:])

    return {"train": train_terms, "test": test_terms}


def main() -> None:
    """Thực thi tạo file phân chia train/test và in thống kê chi tiết."""
    print("=== BẮT ĐẦU CHIA TẬP TRAIN / TEST ===")
    split_data = split_search_terms()

    train_set = set(split_data["train"])
    test_set = set(split_data["test"])
    overlap = train_set & test_set

    unique_count = len(train_set) + len(test_set)
    train_count = len(split_data["train"])
    test_count = len(split_data["test"])
    overlap_count = len(overlap)

    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    md5_hash = compute_file_md5(SPLIT_PATH)

    print(f"Số truy vấn duy nhất: {unique_count}")
    print(f"Số train: {train_count}")
    print(f"Số test: {test_count}")
    print(f"Số truy vấn trùng giữa train và test: {overlap_count}")
    print(f"MD5 của {SPLIT_PATH}: {md5_hash}")
    print("=== HOÀN TẤT PHÂN CHIA TẬP DỮ LIỆU ===")


if __name__ == "__main__":
    main()
