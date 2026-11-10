"""Mô-đun phân tích và xử lý truy vấn tìm kiếm theo SPEC.md mục 8."""

from collections import OrderedDict
from dataclasses import dataclass
import re
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from config import QUERY_CACHE_SIZE
from ingestion.normalize import normalize_query

VI_CHARS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")

_query_cache: "OrderedDict[str, List[float]]" = OrderedDict()


class EmptyQueryError(ValueError):
    """Ngoại lệ khi chuỗi truy vấn rỗng sau chuẩn hóa."""

    pass


@dataclass
class ParsedQuery:
    """Cấu trúc dữ liệu chứa kết quả phân tích truy vấn."""

    raw: str
    normalized: str
    lang: str
    has_model_code: bool
    vector: Optional[List[float]] = None


def detect_language(normalized_text: str) -> str:
    """Nhận diện ngôn ngữ truy vấn dựa trên ký tự tiếng Việt có dấu."""
    if any(c in VI_CHARS for c in normalized_text):
        return "vi"
    return "en"


def check_model_code(normalized_text: str) -> bool:
    """Kiểm tra sự xuất hiện của token mã model dài từ 5 ký tự gồm chữ và số."""
    tokens = re.findall(r"[a-z0-9]+", normalized_text)
    for token in tokens:
        if len(token) >= 5 and any(c.isalpha() for c in token) and any(c.isdigit() for c in token):
            return True
    return False


def encode_query(normalized_text: str) -> Tuple[List[float], bool]:
    """Mã hóa chuỗi truy vấn đã chuẩn hóa thành vector 1024 chiều, chuẩn hóa L2 và dùng cache LRU."""
    global _query_cache
    if normalized_text in _query_cache:
        _query_cache.move_to_end(normalized_text)
        return _query_cache[normalized_text], True

    from indexing.model import get_embedding_model

    model = get_embedding_model()
    vec_np = model.encode(
        [normalized_text],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    vec_np = vec_np.astype(np.float32)

    norm = float(np.linalg.norm(vec_np))
    if norm > 0:
        vec_np = vec_np / norm

    vec: List[float] = vec_np.tolist()
    _query_cache[normalized_text] = vec
    _query_cache.move_to_end(normalized_text)
    if len(_query_cache) > QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)

    return vec, False


def parse(raw_query: str, with_vector: bool = False) -> ParsedQuery:
    """Phân tích chuỗi truy vấn thô thành đối tượng ParsedQuery kèm tùy chọn tạo vector."""
    normalized = normalize_query(raw_query)
    if not normalized:
        raise EmptyQueryError("Truy vấn rỗng sau khi chuẩn hóa.")

    lang = detect_language(normalized)
    has_model_code = check_model_code(normalized)

    vector: Optional[List[float]] = None
    if with_vector:
        vector, _ = encode_query(normalized)

    return ParsedQuery(
        raw=raw_query,
        normalized=normalized,
        lang=lang,
        has_model_code=has_model_code,
        vector=vector,
    )


if __name__ == "__main__":
    queries = sys.argv[1:]
    if not queries:
        print("Sử dụng: python -m query.parser <truy vấn 1> <truy vấn 2> ...")
        sys.exit(0)

    encoded_vectors: List[np.ndarray] = []

    for q in queries:
        parsed = parse(q, with_vector=False)
        t_start = time.perf_counter()
        vec, hit = encode_query(parsed.normalized)
        encode_time_ms = (time.perf_counter() - t_start) * 1000

        vec_arr = np.array(vec, dtype=np.float32)
        encoded_vectors.append(vec_arr)
        l2_norm = float(np.linalg.norm(vec_arr))
        first_3 = [round(float(x), 4) for x in vec[:3]]

        print(f"Truy vấn: '{q}'")
        print(f"  normalized: '{parsed.normalized}'")
        print(f"  lang: {parsed.lang}")
        print(f"  has_model_code: {parsed.has_model_code}")
        print(f"  độ dài vector: {len(vec)}")
        print(f"  chuẩn L2: {l2_norm:.6f}")
        print(f"  3 phần tử đầu: {first_3}")
        print(f"  thời gian encode: {encode_time_ms:.2f} ms")
        print(f"  có trúng cache không: {hit}")
        print()

    # In ma trận cosine giữa các truy vấn
    n_queries = len(queries)
    cos_matrix = np.zeros((n_queries, n_queries), dtype=np.float32)
    for i in range(n_queries):
        for j in range(n_queries):
            cos_matrix[i, j] = float(np.dot(encoded_vectors[i], encoded_vectors[j]))

    print("Ma trận cosine giữa các truy vấn:")
    col_width = 24
    row_fmt = "".join(f"{{:>{col_width}}}" for _ in range(n_queries + 1))
    header = [""] + [f"Q{i+1}: {q[:15]}" for i, q in enumerate(queries)]
    print(row_fmt.format(*header))
    for i in range(n_queries):
        row_vals = [f"Q{i+1}: {queries[i][:15]}"] + [f"{cos_matrix[i, j]:.4f}" for j in range(n_queries)]
        print(row_fmt.format(*row_vals))

    if n_queries >= 3:
        cos_0_1 = cos_matrix[0, 1]
        cos_1_2 = cos_matrix[1, 2]
        print(f"\nSo sánh cosine:")
        print(f"  cosine('{queries[0]}', '{queries[1]}') = {cos_0_1:.4f}")
        print(f"  cosine('{queries[1]}', '{queries[2]}') = {cos_1_2:.4f}")
        print(f"  cosine('{queries[0]}', '{queries[1]}') > cosine('{queries[1]}', '{queries[2]}'): {cos_0_1 > cos_1_2}")
