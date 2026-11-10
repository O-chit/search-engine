"""Kiểm thử chuẩn hóa văn bản và truy vấn."""

from ingestion.normalize import clean_text, normalize_query


def test_clean_text_fused_words() -> None:
    """Kiểm tra tách từ dính liền."""
    assert clean_text("chairsThe table") == "chairs The table"


def test_clean_text_html_tags() -> None:
    """Kiểm tra loại bỏ thẻ HTML và gộp khoảng trắng."""
    assert clean_text("<p>Hello   world</p>") == "Hello world"


def test_clean_text_html_entities() -> None:
    """Kiểm tra giải mã thực thể HTML."""
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"


def test_clean_text_unicode_nfc() -> None:
    """Kiểm tra chuẩn hóa Unicode sang dạng NFC."""
    assert clean_text("cafe\u0301") == "café"


def test_clean_text_brand_split_consequence() -> None:
    """Kiểm tra quy tắc tách từ dính liền đối với DeWalt."""
    assert clean_text("DeWalt drill") == "De Walt drill"


def test_clean_text_unclosed_tags_and_comparisons() -> None:
    """Kiểm tra loại bỏ mảnh thẻ HTML chưa đóng và giữ nguyên phép so sánh."""
    assert clean_text("Great drill<br") == "Great drill"
    assert clean_text("Strong bracket</p") == "Strong bracket"
    assert clean_text("size < 5 in.") == "size < 5 in."


def test_clean_text_empty_and_spaces() -> None:
    """Kiểm tra xử lý chuỗi rỗng và khoảng trắng."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_normalize_query_case_and_whitespace() -> None:
    """Kiểm tra chữ thường, NFC và gộp khoảng trắng cho truy vấn."""
    assert normalize_query("  Máy KHOAN   Pin ") == "máy khoan pin"


def test_normalize_query_length_truncation() -> None:
    """Kiểm tra giới hạn độ dài truy vấn tối đa 200 ký tự."""
    assert len(normalize_query("a" * 500)) == 200
    assert normalize_query("a" * 500) == "a" * 200


def test_normalize_query_empty() -> None:
    """Kiểm tra truy vấn rỗng."""
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""
