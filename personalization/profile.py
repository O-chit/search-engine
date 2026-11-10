"""Quản lý và cập nhật hồ sơ sở thích cá nhân hóa của phiên theo SPEC.md mục 11.2."""

from datetime import datetime, timezone
from typing import Any, List, Optional, Union
import numpy as np

from config import CLICK_DEDUPE_SECONDS, ETA, RECENT_MAX
from indexing.model import get_product_embeddings, get_uid_index
from storage.db import (
    ensure_session,
    get_last_click,
    get_profile_vec,
    get_session,
    insert_event,
    update_session_profile,
)


def update_profile_vec(
    profile_vec: Optional[np.ndarray],
    e: np.ndarray,
    eta: float = ETA,
) -> np.ndarray:
    """Cập nhật vector sở thích trung bình trượt theo công thức EMA và chuẩn hóa độ dài 1."""
    e_vec = np.asarray(e, dtype=np.float32)
    if profile_vec is None:
        norm = float(np.linalg.norm(e_vec))
        if norm > 0:
            return (e_vec / norm).astype(np.float32)
        return e_vec.astype(np.float32)

    p_vec = np.asarray(profile_vec, dtype=np.float32)
    updated = (1.0 - eta) * p_vec + eta * e_vec
    norm = float(np.linalg.norm(updated))
    if norm > 0:
        return (updated / norm).astype(np.float32)
    return updated.astype(np.float32)


def _to_timestamp(val: Union[str, float, int, datetime]) -> float:
    """Chuyển đổi các định dạng thời gian khác nhau thành mốc thời gian dạng số thực giây."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, datetime):
        return val.timestamp()
    if isinstance(val, str):
        clean_str = val.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_str).timestamp()
    return float(val)


def is_valid_click(
    last_product_uid: Optional[str],
    last_ts: Optional[Union[str, float, int, datetime]],
    current_product_uid: str,
    current_ts: Union[str, float, int, datetime],
    dedupe_seconds: float = CLICK_DEDUPE_SECONDS,
) -> bool:
    """Kiểm tra lượt click hợp lệ khi không trùng lặp sản phẩm ngay trước đó trong khoảng thời gian chống trùng."""
    if last_product_uid is None or last_ts is None:
        return True
    if current_product_uid != last_product_uid:
        return True

    t_prev = _to_timestamp(last_ts)
    t_curr = _to_timestamp(current_ts)
    delta = t_curr - t_prev
    if delta < dedupe_seconds:
        return False
    return True


def update_recent_uids(
    recent_uids: List[str],
    product_uid: str,
    max_items: int = RECENT_MAX,
) -> List[str]:
    """Đưa mã sản phẩm mới vào đầu danh sách gần đây, loại bỏ bản ghi trùng và giới hạn độ dài."""
    filtered = [u for u in recent_uids if u != product_uid]
    result = [product_uid] + filtered
    return result[:max_items]


def compute_recent_vec(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """Tính vector trung bình của các sản phẩm đã xem gần nhất và chuẩn hóa L2 về độ dài 1."""
    if not embeddings:
        return None
    stacked = np.stack(embeddings, axis=0).astype(np.float32)
    mean_vec = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0:
        return (mean_vec / norm).astype(np.float32)
    return mean_vec.astype(np.float32)


def apply_click(
    session_id: str,
    product_uid: str,
    ts: Optional[str] = None,
    query_id: Optional[str] = None,
    position: int = 0,
    event_type: str = "click",
) -> bool:
    """Điều phối ghi nhận sự kiện click vào SQLite và cập nhật hồ sơ phiên nếu click hợp lệ."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()

    ensure_session(session_id)
    last_click = get_last_click(session_id)

    insert_event(
        query_id=query_id or "",
        session_id=session_id,
        product_uid=product_uid,
        event_type=event_type,
        position=position,
        ts=ts,
    )

    valid = is_valid_click(
        last_product_uid=last_click["product_uid"] if last_click else None,
        last_ts=last_click["ts"] if last_click else None,
        current_product_uid=product_uid,
        current_ts=ts,
        dedupe_seconds=CLICK_DEDUPE_SECONDS,
    )

    if not valid:
        return False

    all_embeddings = get_product_embeddings()
    uid_index = get_uid_index()
    product_vec: Optional[np.ndarray] = None
    if all_embeddings is not None and uid_index is not None and product_uid in uid_index:
        idx = uid_index[product_uid]
        product_vec = np.asarray(all_embeddings[idx], dtype=np.float32)

    current_profile = get_profile_vec(session_id)
    session_data = get_session(session_id)
    recent_uids = list(session_data.get("recent_uids") or []) if session_data else []
    current_count = int(session_data.get("event_count") or 0) if session_data else 0

    if product_vec is not None:
        new_profile = update_profile_vec(current_profile, product_vec, eta=ETA)
    else:
        new_profile = current_profile

    new_recent = update_recent_uids(recent_uids, product_uid, max_items=RECENT_MAX)
    new_count = current_count + 1

    update_session_profile(
        session_id=session_id,
        profile_vec=new_profile,
        recent_uids=new_recent,
        event_count=new_count,
        ts=ts,
    )
    return True
