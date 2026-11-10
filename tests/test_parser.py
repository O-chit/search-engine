"""Kiểm thử mô-đun phân tích truy vấn query/parser.py."""

import pytest

from query.parser import EmptyQueryError, parse


def test_parser_acceptance_requirements() -> None:
    """Kiểm tra đầy đủ các yêu cầu nghiệm thu đối với hàm parse."""
    assert parse("Máy khoan pin DeWalt 20V").lang == "vi"
    assert parse("cordless drill").lang == "en"
    assert parse("dewalt dcd771c2").has_model_code is True
    assert parse("cordless drill 20v").has_model_code is False

    with pytest.raises((EmptyQueryError, ValueError)):
        parse("   ")


def test_parser_empty_and_whitespace_queries() -> None:
    """Kiểm tra các trường hợp truy vấn rỗng sau chuẩn hóa ném ngoại lệ."""
    with pytest.raises(EmptyQueryError):
        parse("")
    with pytest.raises(EmptyQueryError):
        parse("\t\n  ")


def test_parser_vietnamese_diacritics() -> None:
    """Kiểm tra nhận diện tiếng Việt với các ký tự có dấu khác nhau."""
    assert parse("bóng đèn led").lang == "vi"
    assert parse("sơn tường siêu bóng").lang == "vi"
    assert parse("đinh vít").lang == "vi"


def test_parser_english_queries() -> None:
    """Kiểm tra truy vấn tiếng Anh không dấu."""
    assert parse("water heater 50 gal").lang == "en"
    assert parse("paint brush set").lang == "en"


def test_parser_model_code_detection() -> None:
    """Kiểm tra phát hiện token mã model phức tạp và đơn giản."""
    assert parse("makita xt269m combo kit").has_model_code is True
    assert parse("hammer drill bit").has_model_code is False
    assert parse("12345").has_model_code is False
    assert parse("drill 24v").has_model_code is False
