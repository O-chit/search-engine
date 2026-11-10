"""Endpoint truy xuất và xóa lịch sử sở thích của phiên người dùng."""

from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Response, status

from api.schemas import SessionResponse
from storage.db import clear_session, get_session

router = APIRouter()


@router.get("/session", response_model=SessionResponse)
def get_session_info(
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> SessionResponse:
    """Trả về số lượng sự kiện đã ghi nhận và danh sách mã sản phẩm gần nhất của phiên."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Thiếu header X-Session-Id.")

    session_data = get_session(x_session_id)
    if not session_data:
        return SessionResponse(event_count=0, recent_uids=[])

    return SessionResponse(
        event_count=int(session_data.get("event_count") or 0),
        recent_uids=list(session_data.get("recent_uids") or []),
    )


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def delete_session_info(
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> Response:
    """Xóa bỏ hồ sơ sở thích và lịch sử click của phiên làm việc hiện tại."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Thiếu header X-Session-Id.")

    clear_session(x_session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
