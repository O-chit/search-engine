"""Kiểm thử quy trình pipeline và xử lý lỗi theo SPEC.md mục 9.4."""

import asyncio
from typing import List
from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.pipeline import run_pipeline
from query.parser import ParsedQuery
from retrieval.bm25 import BM25Hit, BM25RetrievalError
from retrieval.vector import VectorHit, VectorRetrievalError

client = TestClient(app)


def dummy_parsed(query_text: str = "drill") -> ParsedQuery:
    """Tạo đối tượng ParsedQuery giả lập để kiểm thử không cần mô hình."""
    return ParsedQuery(
        raw=query_text,
        normalized=query_text,
        lang="en",
        has_model_code=False,
        vector=[0.1] * 1024,
    )


def test_bm25_error_uses_vector_and_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra nhánh BM25 ném lỗi thì kết quả lấy từ nhánh vector và degraded=True."""
    monkeypatch.setattr("api.pipeline.parse", lambda q, with_vector=False: dummy_parsed(q))

    async def mock_search_bm25(query: ParsedQuery, k: int = 200) -> List[BM25Hit]:
        raise BM25RetrievalError("Lỗi kết nối BM25")

    async def mock_search_vector(query: ParsedQuery, k: int = 200) -> List[VectorHit]:
        return [
            VectorHit(
                product_uid="vec_1",
                score=0.95,
                rank=1,
                title="Vector Title 1",
                brand="Brand 1",
            ),
            VectorHit(
                product_uid="vec_2",
                score=0.85,
                rank=2,
                title="Vector Title 2",
                brand="Brand 2",
            ),
        ]

    monkeypatch.setattr("api.pipeline.search_bm25", mock_search_bm25)
    monkeypatch.setattr("api.pipeline.search_vector", mock_search_vector)

    result = asyncio.run(run_pipeline("cordless drill", config_name="hybrid"))
    assert result.degraded is True
    assert len(result.candidates) == 2
    assert result.candidates[0].product_uid == "vec_1"
    assert result.candidates[1].product_uid == "vec_2"


def test_vector_timeout_uses_bm25_and_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra nhánh vector quá hạn thì kết quả lấy từ BM25 và degraded=True."""
    monkeypatch.setattr("api.pipeline.parse", lambda q, with_vector=False: dummy_parsed(q))

    async def mock_search_bm25(query: ParsedQuery, k: int = 200) -> List[BM25Hit]:
        return [
            BM25Hit(
                product_uid="bm25_1",
                score=15.0,
                rank=1,
                title="BM25 Title 1",
                brand="Brand 1",
            )
        ]

    async def mock_search_vector_timeout(query: ParsedQuery, k: int = 200) -> List[VectorHit]:
        await asyncio.sleep(0.5)
        return []

    monkeypatch.setattr("api.pipeline.search_bm25", mock_search_bm25)
    monkeypatch.setattr("api.pipeline.search_vector", mock_search_vector_timeout)

    result = asyncio.run(run_pipeline("cordless drill", config_name="hybrid"))
    assert result.degraded is True
    assert len(result.candidates) == 1
    assert result.candidates[0].product_uid == "bm25_1"


def test_both_branches_fail_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra cả hai nhánh tìm kiếm đều lỗi thì API trả mã trạng thái 503."""
    monkeypatch.setattr("api.pipeline.parse", lambda q, with_vector=False: dummy_parsed(q))

    async def mock_search_bm25(query: ParsedQuery, k: int = 200) -> List[BM25Hit]:
        raise BM25RetrievalError("BM25 chết hoàn toàn")

    async def mock_search_vector(query: ParsedQuery, k: int = 200) -> List[VectorHit]:
        raise VectorRetrievalError("Vector chết hoàn toàn")

    monkeypatch.setattr("api.pipeline.search_bm25", mock_search_bm25)
    monkeypatch.setattr("api.pipeline.search_vector", mock_search_vector)

    response = client.get("/search?q=hammer")
    assert response.status_code == 503


def test_no_candidates_returns_200_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra không có ứng viên nào thì API trả mã 200 kèm danh sách rỗng."""
    monkeypatch.setattr("api.pipeline.parse", lambda q, with_vector=False: dummy_parsed(q))

    async def mock_search_bm25(query: ParsedQuery, k: int = 200) -> List[BM25Hit]:
        return []

    async def mock_search_vector(query: ParsedQuery, k: int = 200) -> List[VectorHit]:
        return []

    monkeypatch.setattr("api.pipeline.search_bm25", mock_search_bm25)
    monkeypatch.setattr("api.pipeline.search_vector", mock_search_vector)

    response = client.get("/search?q=unobtainium_tool_12345")
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["total"] == 0
    assert data["degraded"] is False
