"""Mô-đun chuẩn hóa văn bản dùng chung cho dữ liệu lưu trữ và truy vấn."""

import html
import re
import unicodedata

from config import MAX_QUERY_CHARS

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_UNCLOSED_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*\s*$")
_FUSED_WORD_RE = re.compile(r"([a-z])([A-Z])(?=[a-z])")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Làm sạch văn bản lưu trữ bằng cách bỏ HTML, giải mã thực thể, chuẩn hóa NFC và gộp khoảng trắng."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = _UNCLOSED_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _FUSED_WORD_RE.sub(r"\1 \2", text)
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_query(text: str) -> str:
    """Chuẩn hóa truy vấn người dùng: NFC, chữ thường, gộp khoảng trắng và cắt độ dài tối đa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_QUERY_CHARS]
