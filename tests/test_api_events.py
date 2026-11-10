"""Kiểm thử API cho sự kiện, phiên và tìm kiếm theo SPEC.md mục 15."""

from typing import List
import uuid
from fastapi.testclient import TestClient
import pytest

from api.main import app
from query.parser import ParsedQuery
from retrieval.bm25 import BM25Hit
from retrieval.vector import VectorHit
from storage.db import record_query

# Làm nóng kết nối OpenSearch để tránh quá hạn 150ms ở truy vấn đầu
try:
    from retrieval.client import get_opensearch_client
    _os_client = get_opensearch_client()
    _os_client.search(
        index="products",
        body={"size": 1, "query": {"knn": {"embedding": {"vector": [0.0] * 1024, "k": 1}}}},
        request_timeout=10.0,
    )
except Exception:
    pass

client = TestClient(app)


def test_search_missing_q_returns_422() -> None:
    """Kiểm tra gọi /search thiếu tham số bắt buộc q trả về mã 422."""
    response = client.get("/search")
    assert response.status_code == 422


def test_search_missing_session_header_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra gọi /search thiếu header X-Session-Id vẫn trả 200 và có danh sách kết quả."""
    dummy = ParsedQuery(
        raw="drill",
        normalized="drill",
        lang="en",
        has_model_code=False,
        vector=[0.1] * 1024,
    )
    monkeypatch.setattr("search.pipeline.parse", lambda q, with_vector=False: dummy)

    async def mock_bm25(query: ParsedQuery, k: int = 200) -> List[BM25Hit]:
        return [
            BM25Hit(
                product_uid="100001",
                score=10.0,
                rank=1,
                title="Cordless Drill",
                brand="DeWalt",
                bullets="Great drill",
                description="Power tool",
            )
        ]

    async def mock_vec(query: ParsedQuery, k: int = 200) -> List[VectorHit]:
        return [
            VectorHit(
                product_uid="100001",
                score=0.9,
                rank=1,
                title="Cordless Drill",
                brand="DeWalt",
                bullets="Great drill",
                description="Power tool",
            )
        ]

    monkeypatch.setattr("search.pipeline.search_bm25", mock_bm25)
    monkeypatch.setattr("search.pipeline.search_vector", mock_vec)

    response = client.get("/search?q=drill")
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["product_uid"] == "100001"


def test_events_query_id_not_found_returns_404() -> None:
    """Kiểm tra gửi sự kiện với query_id không tồn tại trong bảng queries trả về mã 404."""
    fake_query_id = f"q_nonexistent_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/events",
        headers={"X-Session-Id": "test_session_404"},
        json={
            "query_id": fake_query_id,
            "product_uid": "100001",
            "type": "click",
            "position": 1,
        },
    )
    assert response.status_code == 404


def test_events_invalid_type_returns_422() -> None:
    """Kiểm tra gửi sự kiện với type khác 'click' (như 'view') trả về mã 422 từ Pydantic."""
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    record_query(
        query_id=query_id,
        session_id="test_session_type",
        raw_text="hammer",
        normalized_text="hammer",
        lang="en",
        result_uids=["100001"],
        personalized=False,
        degraded=False,
        timing_ms={"total": 10.0},
    )

    response = client.post(
        "/events",
        headers={"X-Session-Id": "test_session_type"},
        json={
            "query_id": query_id,
            "product_uid": "100001",
            "type": "view",
            "position": 1,
        },
    )
    assert response.status_code == 422


def test_session_missing_header_returns_400() -> None:
    """Kiểm tra gọi GET /session và DELETE /session khi thiếu header X-Session-Id trả về mã 400."""
    get_res = client.get("/session")
    assert get_res.status_code == 400

    del_res = client.delete("/session")
    assert del_res.status_code == 400


def test_valid_events_flow_and_session_lifecycle() -> None:
    """Kiểm tra toàn bộ luồng ghi nhận sự kiện click hợp lệ và vòng đời phiên."""
    session_id = f"test_lifecycle_{uuid.uuid4().hex}"
    query_id = f"q_{uuid.uuid4().hex[:8]}"

    record_query(
        query_id=query_id,
        session_id=session_id,
        raw_text="cordless drill",
        normalized_text="cordless drill",
        lang="en",
        result_uids=["100001", "100002"],
        personalized=True,
        degraded=False,
        timing_ms={"total": 25.0},
    )

    # Gửi sự kiện click hợp lệ
    res = client.post(
        "/events",
        headers={"X-Session-Id": session_id},
        json={
            "query_id": query_id,
            "product_uid": "100001",
            "type": "click",
            "position": 1,
        },
    )
    assert res.status_code == 202

    # Kiểm tra GET /session
    sess_res = client.get("/session", headers={"X-Session-Id": session_id})
    assert sess_res.status_code == 200
    sess_data = sess_res.json()
    assert sess_data["event_count"] >= 1
    assert "100001" in sess_data["recent_uids"]

    # DELETE /session
    del_res = client.delete("/session", headers={"X-Session-Id": session_id})
    assert del_res.status_code == 204

    # GET /session sau xóa
    sess_res_after = client.get("/session", headers={"X-Session-Id": session_id})
    assert sess_res_after.status_code == 200
    sess_data_after = sess_res_after.json()
    assert sess_data_after["event_count"] == 0
    assert sess_data_after["recent_uids"] == []
