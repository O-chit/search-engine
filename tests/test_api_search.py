"""Kiểm thử tích hợp các endpoint API /health và /search."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Kiểm tra endpoint /health trả về trạng thái hợp lệ."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "opensearch" in data
    assert "document_count" in data
    assert data["model_loaded"] is False


def test_search_missing_query_returns_422() -> None:
    """Kiểm tra thiếu tham số q trả về mã lỗi 422."""
    response = client.get("/search")
    assert response.status_code == 422


def test_search_whitespace_query_returns_400() -> None:
    """Kiểm tra truy vấn chỉ có khoảng trắng trả về mã lỗi 400."""
    response = client.get("/search?q=%20%20")
    assert response.status_code == 400


def test_search_normal_query() -> None:
    """Kiểm tra truy vấn tìm kiếm bình thường trả về danh sách kết quả và timing."""
    response = client.get("/search?q=cordless%20drill&size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_query"] == "cordless drill"
    assert data["lang"] == "en"
    assert "timing_ms" in data
    assert "total" in data
    assert isinstance(data["results"], list)


def test_search_pagination_positions() -> None:
    """Kiểm tra phân trang trả về đúng position tăng dần."""
    response = client.get("/search?q=cordless%20drill&page=2&size=5")
    assert response.status_code == 200
    data = response.json()
    if len(data["results"]) > 0:
        expected_positions = list(range(6, 6 + len(data["results"])))
        actual_positions = [item["position"] for item in data["results"]]
        assert actual_positions == expected_positions


def test_search_debug_mode() -> None:
    """Kiểm tra bật tắt cờ debug trong phản hồi."""
    resp_no_debug = client.get("/search?q=drill&size=2&debug=false")
    assert resp_no_debug.status_code == 200
    items_no_debug = resp_no_debug.json()["results"]
    if items_no_debug:
        assert "debug" not in items_no_debug[0]

    resp_debug = client.get("/search?q=drill&size=2&debug=true")
    assert resp_debug.status_code == 200
    items_debug = resp_debug.json()["results"]
    if items_debug:
        assert "debug" in items_debug[0]
        assert "bm25_rank" in items_debug[0]["debug"]
        assert "bm25_score" in items_debug[0]["debug"]
        assert "rel" in items_debug[0]["debug"]
