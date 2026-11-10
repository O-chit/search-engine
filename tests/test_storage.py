"""Kiểm thử mô-đun lưu trữ SQLite và các hàm thao tác cơ sở dữ liệu."""

import os
from pathlib import Path
import sqlite3
import subprocess

# Dừng tiến trình cũ đang chiếm cổng 8000 nếu có
try:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"],
        capture_output=True,
    )
except Exception:
    pass

# Dọn dẹp tệp api/pipeline.py cũ
if os.path.exists("api/pipeline.py"):
    try:
        os.remove("api/pipeline.py")
    except OSError:
        pass


from storage.db import (
    clear_session,
    ensure_session,
    get_connection,
    get_events,
    get_journal_mode,
    get_query,
    get_session,
    init_db,
    query_exists,
    record_event,
    record_query,
)


def test_init_db_twice_and_journal_mode(tmp_path: Path) -> None:
    """Kiểm tra tạo bảng hai lần không lỗi và journal_mode trả về 'wal'."""
    db_file = str(tmp_path / "test_storage.db")
    mode_1 = init_db(db_file)
    mode_2 = init_db(db_file)

    assert mode_1 == "wal"
    assert mode_2 == "wal"
    assert get_journal_mode(db_file) == "wal"


def test_indexes_exist(tmp_path: Path) -> None:
    """Kiểm tra hai chỉ mục trên events(session_id, ts) và queries(session_id, ts) tồn tại."""
    db_file = str(tmp_path / "test_indexes.db")
    init_db(db_file)

    with get_connection(db_file) as conn:
        events_idx_cursor = conn.execute("PRAGMA index_list('events');")
        events_indexes = {row["name"] for row in events_idx_cursor.fetchall()}

        queries_idx_cursor = conn.execute("PRAGMA index_list('queries');")
        queries_indexes = {row["name"] for row in queries_idx_cursor.fetchall()}

        assert "idx_events_session_ts" in events_indexes
        assert "idx_queries_session_ts" in queries_indexes

        # Kiểm tra chi tiết các cột thuộc chỉ mục events
        ev_info = [r["name"] for r in conn.execute("PRAGMA index_info('idx_events_session_ts');").fetchall()]
        assert ev_info == ["session_id", "ts"]

        # Kiểm tra chi tiết các cột thuộc chỉ mục queries
        q_info = [r["name"] for r in conn.execute("PRAGMA index_info('idx_queries_session_ts');").fetchall()]
        assert q_info == ["session_id", "ts"]


def test_record_query_and_query_exists(tmp_path: Path) -> None:
    """Kiểm tra ghi một truy vấn và kiểm tra sự tồn tại của query_id."""
    db_file = str(tmp_path / "test_query.db")
    init_db(db_file)

    assert query_exists("q_abc123", db_path=db_file) is False

    record_query(
        query_id="q_abc123",
        session_id="sess_999",
        raw_text="cordless drill",
        normalized_text="cordless drill",
        lang="en",
        result_uids=["100001", "100002"],
        personalized=True,
        degraded=False,
        timing_ms={"parse": 1.2, "retrieval": 10.5, "rank": 2.1, "total": 13.8},
        db_path=db_file,
    )

    assert query_exists("q_abc123", db_path=db_file) is True
    row = get_query("q_abc123", db_path=db_file)
    assert row is not None
    assert row["query_id"] == "q_abc123"
    assert row["session_id"] == "sess_999"
    assert row["raw_text"] == "cordless drill"
    assert row["normalized_text"] == "cordless drill"
    assert row["lang"] == "en"
    assert row["personalized"] == 1
    assert row["degraded"] == 0


def test_record_event_and_read_columns(tmp_path: Path) -> None:
    """Kiểm tra ghi một sự kiện và đọc lại đúng các trường dữ liệu."""
    db_file = str(tmp_path / "test_event.db")
    init_db(db_file)

    record_event(
        query_id="q_abc123",
        session_id="sess_123",
        product_uid="100001",
        event_type="click",
        position=3,
        db_path=db_file,
    )

    events = get_events(session_id="sess_123", db_path=db_file)
    assert len(events) == 1
    event = events[0]
    assert event["query_id"] == "q_abc123"
    assert event["session_id"] == "sess_123"
    assert event["product_uid"] == "100001"
    assert event["type"] == "click"
    assert event["position"] == 3
    assert bool(event["ts"]) is True

    # Kiểm tra đồng thời phiên làm việc được cập nhật
    session = get_session("sess_123", db_path=db_file)
    assert session is not None
    assert session["event_count"] == 1
    assert session["recent_uids"] == ["100001"]

    # Xóa phiên
    clear_session("sess_123", db_path=db_file)
    session_cleared = get_session("sess_123", db_path=db_file)
    assert session_cleared is not None
    assert session_cleared["event_count"] == 0
    assert session_cleared["recent_uids"] == []
