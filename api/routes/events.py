"""Endpoint ghi nhận sự kiện tương tác của người dùng như click sản phẩm."""

from typing import Dict, Optional
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, status

from api.schemas import EventCreateRequest
from storage.db import query_exists, record_event

router = APIRouter()


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
def create_event(
    event_data: EventCreateRequest,
    background_tasks: BackgroundTasks,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> Dict[str, str]:
    """Ghi nhận sự kiện click của người dùng trong phiên làm việc qua cơ chế tác vụ nền."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Thiếu header X-Session-Id.")
    if not query_exists(event_data.query_id):
        raise HTTPException(
            status_code=404,
            detail=f"Mã truy vấn '{event_data.query_id}' không tồn tại trong hệ thống.",
        )

    background_tasks.add_task(
        record_event,
        query_id=event_data.query_id,
        session_id=x_session_id,
        product_uid=event_data.product_uid,
        event_type=event_data.type,
        position=event_data.position,
    )
    return {"status": "accepted"}
