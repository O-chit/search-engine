"""Mô-đun phân tích và xử lý truy vấn tìm kiếm."""

from dataclasses import dataclass
import re
from typing import List, Optional

from ingestion.normalize import normalize_query

VI_CHARS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


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


def parse(raw_query: str) -> ParsedQuery:
    """Phân tích chuỗi truy vấn thô thành đối tượng ParsedQuery."""
    normalized = normalize_query(raw_query)
    if not normalized:
        raise EmptyQueryError("Truy vấn rỗng sau khi chuẩn hóa.")

    lang = detect_language(normalized)
    has_model_code = check_model_code(normalized)

    return ParsedQuery(
        raw=raw_query,
        normalized=normalized,
        lang=lang,
        has_model_code=has_model_code,
        vector=None,
    )
