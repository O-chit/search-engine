"""Mô-đun quản lý lưu trữ SQLite cho phiên, truy vấn và sự kiện theo SPEC.md mục 12.1."""

from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from config import RECENT_MAX, SQLITE_PATH


def get_connection(db_path: str = SQLITE_PATH) -> sqlite3.Connection:
    """Tạo kết nối SQLite với cấu hình WAL và trả về đối tượng Connection."""
    dir_name = os.path.dirname(os.path.abspath(db_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str = SQLITE_PATH) -> str:
    """Khởi tạo các bảng và chỉ mục trong cơ sở dữ liệu SQLite nếu chưa tồn tại."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("PRAGMA journal_mode=WAL;")
        row = cursor.fetchone()
        journal_mode = str(row[0]).lower() if row else "unknown"

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                profile_vec BLOB,
                recent_uids TEXT,
                event_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                session_id TEXT,
                raw_text TEXT,
                normalized_text TEXT,
                lang TEXT,
                result_uids TEXT,
                personalized INTEGER,
                degraded INTEGER,
                timing_ms TEXT,
                ts TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                session_id TEXT,
                product_uid TEXT,
                type TEXT,
                position INTEGER,
                ts TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_queries_session_ts ON queries(session_id, ts);"
        )
        conn.commit()
    return journal_mode


def get_journal_mode(db_path: str = SQLITE_PATH) -> str:
    """Trả về chế độ journal_mode hiện tại của tệp cơ sở dữ liệu SQLite."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("PRAGMA journal_mode;")
        row = cursor.fetchone()
        return str(row[0]).lower() if row else "unknown"


def ensure_session(session_id: str, db_path: str = SQLITE_PATH) -> None:
    """Đảm bảo dòng thông tin phiên tồn tại trong bảng sessions."""
    if not session_id:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, profile_vec, recent_uids, event_count, created_at, updated_at)
            VALUES (?, NULL, '[]', 0, ?, ?);
            """,
            (session_id, now_iso, now_iso),
        )
        conn.commit()


def record_query(
    query_id: str,
    session_id: Optional[str],
    raw_text: str,
    normalized_text: str,
    lang: str,
    result_uids: List[str],
    personalized: bool,
    degraded: bool,
    timing_ms: Dict[str, Any],
    ts: Optional[str] = None,
    db_path: str = SQLITE_PATH,
) -> None:
    """Ghi nhận thông tin một lượt truy vấn tìm kiếm vào bảng queries."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    result_uids_json = json.dumps(result_uids, ensure_ascii=False)
    timing_ms_json = json.dumps(timing_ms, ensure_ascii=False)
    personalized_val = 1 if personalized else 0
    degraded_val = 1 if degraded else 0

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO queries (
                query_id, session_id, raw_text, normalized_text, lang,
                result_uids, personalized, degraded, timing_ms, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                query_id,
                session_id,
                raw_text,
                normalized_text,
                lang,
                result_uids_json,
                personalized_val,
                degraded_val,
                timing_ms_json,
                ts,
            ),
        )
        conn.commit()


def query_exists(query_id: str, db_path: str = SQLITE_PATH) -> bool:
    """Kiểm tra sự tồn tại của query_id trong bảng queries."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM queries WHERE query_id = ? LIMIT 1;",
            (query_id,),
        )
        return cursor.fetchone() is not None


def get_query(query_id: str, db_path: str = SQLITE_PATH) -> Optional[Dict[str, Any]]:
    """Đọc thông tin chi tiết một truy vấn theo query_id từ bảng queries."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT query_id, session_id, raw_text, normalized_text, lang,
                   result_uids, personalized, degraded, timing_ms, ts
            FROM queries WHERE query_id = ? LIMIT 1;
            """,
            (query_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)


def record_event(
    query_id: str,
    session_id: str,
    product_uid: str,
    event_type: str,
    position: int,
    ts: Optional[str] = None,
    db_path: str = SQLITE_PATH,
) -> None:
    """Ghi nhận sự kiện click sản phẩm và cập nhật danh sách recent_uids của phiên."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO events (query_id, session_id, product_uid, type, position, ts)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (query_id, session_id, product_uid, event_type, position, ts),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, profile_vec, recent_uids, event_count, created_at, updated_at)
            VALUES (?, NULL, '[]', 0, ?, ?);
            """,
            (session_id, ts, ts),
        )
        cursor = conn.execute(
            "SELECT event_count, recent_uids FROM sessions WHERE session_id = ?;",
            (session_id,),
        )
        row = cursor.fetchone()
        current_count = int(row["event_count"] or 0) if row else 0
        raw_uids = row["recent_uids"] if row and row["recent_uids"] else "[]"
        try:
            uids = json.loads(raw_uids)
            if not isinstance(uids, list):
                uids = []
        except Exception:
            uids = []

        new_recent = [product_uid] + [u for u in uids if u != product_uid]
        new_recent = new_recent[:RECENT_MAX]
        new_count = current_count + 1

        conn.execute(
            """
            UPDATE sessions
            SET event_count = ?, recent_uids = ?, updated_at = ?
            WHERE session_id = ?;
            """,
            (new_count, json.dumps(new_recent, ensure_ascii=False), ts, session_id),
        )
        conn.commit()


def get_session(session_id: str, db_path: str = SQLITE_PATH) -> Optional[Dict[str, Any]]:
    """Đọc dữ liệu phiên làm việc từ bảng sessions."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT session_id, profile_vec, recent_uids, event_count, created_at, updated_at
            FROM sessions WHERE session_id = ? LIMIT 1;
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        res = dict(row)
        try:
            res["recent_uids"] = json.loads(res["recent_uids"] or "[]")
        except Exception:
            res["recent_uids"] = []
        return res


def clear_session(session_id: str, db_path: str = SQLITE_PATH) -> None:
    """Xóa hồ sơ vector và lịch sử click của một phiên làm việc."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, profile_vec, recent_uids, event_count, created_at, updated_at)
            VALUES (?, NULL, '[]', 0, ?, ?);
            """,
            (session_id, now_iso, now_iso),
        )
        conn.execute(
            """
            UPDATE sessions
            SET profile_vec = NULL, recent_uids = '[]', event_count = 0, updated_at = ?
            WHERE session_id = ?;
            """,
            (now_iso, session_id),
        )
        conn.commit()


def get_events(session_id: Optional[str] = None, db_path: str = SQLITE_PATH) -> List[Dict[str, Any]]:
    """Lấy danh sách các sự kiện đã ghi nhận theo phiên hoặc toàn bộ."""
    with get_connection(db_path) as conn:
        if session_id:
            cursor = conn.execute(
                """
                SELECT event_id, query_id, session_id, product_uid, type, position, ts
                FROM events WHERE session_id = ? ORDER BY event_id ASC;
                """,
                (session_id,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT event_id, query_id, session_id, product_uid, type, position, ts
                FROM events ORDER BY event_id ASC;
                """
            )
        return [dict(r) for r in cursor.fetchall()]
