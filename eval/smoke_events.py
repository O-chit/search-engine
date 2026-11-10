"""Kịch bản kiểm thử khói (smoke test) các endpoint sự kiện, phiên và lưu trữ SQLite."""

import json
import sys
import time
from typing import Any, Dict, List, Tuple
import uuid

import httpx

from config import API_PORT, SQLITE_PATH
from storage.db import (
    get_connection,
    get_events,
    get_journal_mode,
    get_query,
    get_session,
)


def run_smoke_tests() -> bool:
    """Thực thi 14 bước kiểm thử khói và in bảng kết quả chi tiết."""
    base_url = f"http://127.0.0.1:{API_PORT}"
    session_id = f"smoke_sess_{uuid.uuid4().hex[:12]}"
    test_results: List[Tuple[str, str, str]] = []

    client = httpx.Client(base_url=base_url, timeout=10.0)

    # 1. GET / trả 200, có đủ 8 id bắt buộc
    try:
        r1 = client.get("/")
        html = r1.text
        required_ids = [
            "q",
            "search-btn",
            "results",
            "toggle-personalize",
            "toggle-debug",
            "clear-history",
            "timing",
            "click-count",
        ]
        missing_ids = [id_name for id_name in required_ids if f'id="{id_name}"' not in html and f"id='{id_name}'" not in html]
        if r1.status_code == 200 and not missing_ids:
            test_results.append((
                "(1) GET / trả 200, có đủ 8 id bắt buộc",
                "PASS",
                f"200 OK, đủ 8 id: {required_ids}",
            ))
        else:
            test_results.append((
                "(1) GET / trả 200, có đủ 8 id bắt buộc",
                "FAIL",
                f"Mã: {r1.status_code}, thiếu id: {missing_ids}",
            ))
    except Exception as exc:
        test_results.append(("(1) GET / trả 200, có đủ 8 id bắt buộc", "FAIL", str(exc)))

    # 2. GET /web/app.js trả 200, nội dung có 'localStorage', 'X-Session-Id', '/events', 'DELETE'
    try:
        r2 = client.get("/web/app.js")
        js_text = r2.text
        keywords = ["localStorage", "X-Session-Id", "/events", "DELETE"]
        missing_kw = [kw for kw in keywords if kw not in js_text]
        if r2.status_code == 200 and not missing_kw:
            test_results.append((
                "(2) GET /web/app.js trả 200, nội dung có 'localStorage', 'X-Session-Id', '/events', 'DELETE'",
                "PASS",
                f"200 OK, chứa đủ {keywords}",
            ))
        else:
            test_results.append((
                "(2) GET /web/app.js trả 200, nội dung có 'localStorage', 'X-Session-Id', '/events', 'DELETE'",
                "FAIL",
                f"Mã: {r2.status_code}, thiếu từ khóa: {missing_kw}",
            ))
    except Exception as exc:
        test_results.append(("(2) GET /web/app.js trả 200", "FAIL", str(exc)))

    query_id_step3 = None
    results_step3 = []

    # 3. /search 'cordless drill' page=1 kèm header phiên mới: 200, có query_id
    try:
        r3 = client.get(
            "/search?q=cordless%20drill&page=1",
            headers={"X-Session-Id": session_id},
        )
        data3 = r3.json() if r3.status_code == 200 else {}
        query_id_step3 = data3.get("query_id")
        results_step3 = data3.get("results", [])
        if r3.status_code == 200 and query_id_step3 and len(results_step3) >= 3:
            test_results.append((
                "(3) /search 'cordless drill' page=1 kèm header phiên mới: 200, có query_id",
                "PASS",
                f"200 OK, query_id={query_id_step3}, results={len(results_step3)}",
            ))
        else:
            test_results.append((
                "(3) /search 'cordless drill' page=1 kèm header phiên mới: 200, có query_id",
                "FAIL",
                f"Mã: {r3.status_code}, query_id={query_id_step3}",
            ))
    except Exception as exc:
        test_results.append(("(3) /search page=1 có header phiên", "FAIL", str(exc)))

    # 4. dòng sessions của phiên mới tồn tại
    try:
        sess_row = get_session(session_id)
        if sess_row and sess_row["session_id"] == session_id:
            test_results.append((
                "(4) dòng sessions của phiên mới tồn tại",
                "PASS",
                f"Tồn tại session_id={session_id}, event_count={sess_row['event_count']}",
            ))
        else:
            test_results.append((
                "(4) dòng sessions của phiên mới tồn tại",
                "FAIL",
                f"Không tìm thấy phiên {session_id} trong SQLite",
            ))
    except Exception as exc:
        test_results.append(("(4) dòng sessions của phiên mới tồn tại", "FAIL", str(exc)))

    # 5. dòng queries của query_id đó có session_id đúng, result_uids đúng bằng danh sách product_uid trả về theo đúng thứ tự, personalized và degraded là 0/1, timing_ms là JSON hợp lệ
    try:
        q_row = get_query(query_id_step3) if query_id_step3 else None
        if not q_row:
            test_results.append((
                "(5) dòng queries của query_id đó có dữ liệu đúng chuẩn",
                "FAIL",
                f"Không tìm thấy query_id={query_id_step3} trong bảng queries",
            ))
        else:
            actual_uids = json.loads(q_row["result_uids"])
            expected_uids = [item["product_uid"] for item in results_step3]
            timing_obj = json.loads(q_row["timing_ms"])

            is_sess_match = q_row["session_id"] == session_id
            is_uids_match = actual_uids == expected_uids
            is_personalized_valid = q_row["personalized"] in (0, 1)
            is_degraded_valid = q_row["degraded"] in (0, 1)
            is_timing_valid = isinstance(timing_obj, dict) and "total" in timing_obj

            if is_sess_match and is_uids_match and is_personalized_valid and is_degraded_valid and is_timing_valid:
                test_results.append((
                    "(5) dòng queries của query_id đó khớp cấu trúc và giá trị",
                    "PASS",
                    f"uids={len(actual_uids)}, pers={q_row['personalized']}, deg={q_row['degraded']}",
                ))
            else:
                test_results.append((
                    "(5) dòng queries của query_id đó khớp cấu trúc và giá trị",
                    "FAIL",
                    f"sess={is_sess_match}, uids={is_uids_match}, pers={is_personalized_valid}, deg={is_degraded_valid}, timing={is_timing_valid}",
                ))
    except Exception as exc:
        test_results.append(("(5) dòng queries của query_id đó", "FAIL", str(exc)))

    # 6. gọi lại cùng truy vấn với page=2: số dòng queries của phiên KHÔNG tăng
    try:
        with get_connection() as conn:
            cnt_before = conn.execute(
                "SELECT COUNT(*) FROM queries WHERE session_id = ?;",
                (session_id,),
            ).fetchone()[0]

        r6 = client.get(
            "/search?q=cordless%20drill&page=2",
            headers={"X-Session-Id": session_id},
        )

        with get_connection() as conn:
            cnt_after = conn.execute(
                "SELECT COUNT(*) FROM queries WHERE session_id = ?;",
                (session_id,),
            ).fetchone()[0]

        if r6.status_code == 200 and cnt_after == cnt_before:
            test_results.append((
                "(6) gọi lại cùng truy vấn với page=2: số dòng queries của phiên KHÔNG tăng",
                "PASS",
                f"Trước: {cnt_before}, sau: {cnt_after} (giữ nguyên)",
            ))
        else:
            test_results.append((
                "(6) gọi lại cùng truy vấn với page=2: số dòng queries của phiên KHÔNG tăng",
                "FAIL",
                f"Mã: {r6.status_code}, trước: {cnt_before}, sau: {cnt_after}",
            ))
    except Exception as exc:
        test_results.append(("(6) page=2 số dòng queries không tăng", "FAIL", str(exc)))

    # 7. /search thiếu header: 200, có kết quả, dòng queries mới có session_id NULL
    try:
        r7 = client.get("/search?q=cordless%20drill&page=1")
        data7 = r7.json() if r7.status_code == 200 else {}
        qid7 = data7.get("query_id")
        q7_row = get_query(qid7) if qid7 else None
        if r7.status_code == 200 and q7_row and q7_row["session_id"] is None:
            test_results.append((
                "(7) /search thiếu header: 200, có kết quả, dòng queries mới có session_id NULL",
                "PASS",
                f"200 OK, query_id={qid7}, session_id=NULL",
            ))
        else:
            test_results.append((
                "(7) /search thiếu header: 200, có kết quả, dòng queries mới có session_id NULL",
                "FAIL",
                f"Mã: {r7.status_code}, q_row: {q7_row}",
            ))
    except Exception as exc:
        test_results.append(("(7) search thiếu header", "FAIL", str(exc)))

    # 8. POST /events click sản phẩm ở position 3 của kết quả bước 3: trả 202, sau tối đa 2 giây có đúng 1 dòng events với đúng query_id, session_id, product_uid, type 'click', position 3
    try:
        target_uid = results_step3[2]["product_uid"] if len(results_step3) >= 3 else "unknown"
        r8 = client.post(
            "/events",
            headers={"X-Session-Id": session_id},
            json={
                "query_id": query_id_step3,
                "product_uid": target_uid,
                "type": "click",
                "position": 3,
            },
        )
        event_matched = False
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 2.0:
            events = get_events(session_id)
            if len(events) == 1:
                ev = events[0]
                if (
                    ev["query_id"] == query_id_step3
                    and ev["session_id"] == session_id
                    and ev["product_uid"] == target_uid
                    and ev["type"] == "click"
                    and ev["position"] == 3
                ):
                    event_matched = True
                    break
            time.sleep(0.05)

        elapsed = round(time.perf_counter() - t_start, 2)
        if r8.status_code == 202 and event_matched:
            test_results.append((
                "(8) POST /events click sản phẩm position 3 ghi đúng dữ liệu sự kiện",
                "PASS",
                f"202 Accepted, 1 dòng event sau {elapsed}s",
            ))
        else:
            test_results.append((
                "(8) POST /events click sản phẩm position 3 ghi đúng dữ liệu sự kiện",
                "FAIL",
                f"Mã: {r8.status_code}, matched: {event_matched}, elapsed: {elapsed}s",
            ))
    except Exception as exc:
        test_results.append(("(8) POST /events hợp lệ", "FAIL", str(exc)))

    # 9. POST /events query_id 'q_khong_ton_tai': 404
    try:
        r9 = client.post(
            "/events",
            headers={"X-Session-Id": session_id},
            json={
                "query_id": "q_khong_ton_tai",
                "product_uid": "100001",
                "type": "click",
                "position": 1,
            },
        )
        if r9.status_code == 404:
            test_results.append((
                "(9) POST /events query_id 'q_khong_ton_tai': 404",
                "PASS",
                "404 Not Found",
            ))
        else:
            test_results.append((
                "(9) POST /events query_id 'q_khong_ton_tai': 404",
                "FAIL",
                f"Mã: {r9.status_code}",
            ))
    except Exception as exc:
        test_results.append(("(9) POST /events query_id không tồn tại", "FAIL", str(exc)))

    # 10. POST /events type 'view': 422
    try:
        r10 = client.post(
            "/events",
            headers={"X-Session-Id": session_id},
            json={
                "query_id": query_id_step3,
                "product_uid": "100001",
                "type": "view",
                "position": 1,
            },
        )
        if r10.status_code == 422:
            test_results.append((
                "(10) POST /events type 'view': 422",
                "PASS",
                "422 Unprocessable Entity",
            ))
        else:
            test_results.append((
                "(10) POST /events type 'view': 422",
                "FAIL",
                f"Mã: {r10.status_code}",
            ))
    except Exception as exc:
        test_results.append(("(10) POST /events type 'view'", "FAIL", str(exc)))

    # 11. GET /session: event_count 1, recent_uids đúng [product_uid đã click]
    try:
        r11 = client.get("/session", headers={"X-Session-Id": session_id})
        data11 = r11.json() if r11.status_code == 200 else {}
        cnt11 = data11.get("event_count")
        uids11 = data11.get("recent_uids")
        if r11.status_code == 200 and cnt11 == 1 and uids11 == [target_uid]:
            test_results.append((
                "(11) GET /session: event_count 1, recent_uids đúng [product_uid đã click]",
                "PASS",
                f"event_count=1, recent_uids={uids11}",
            ))
        else:
            test_results.append((
                "(11) GET /session: event_count 1, recent_uids đúng [product_uid đã click]",
                "FAIL",
                f"Mã: {r11.status_code}, event_count={cnt11}, recent_uids={uids11}",
            ))
    except Exception as exc:
        test_results.append(("(11) GET /session", "FAIL", str(exc)))

    # 12. DELETE /session: 204; GET /session sau đó: event_count 0 và recent_uids []
    try:
        r12_del = client.delete("/session", headers={"X-Session-Id": session_id})
        r12_get = client.get("/session", headers={"X-Session-Id": session_id})
        data12 = r12_get.json() if r12_get.status_code == 200 else {}
        cnt12 = data12.get("event_count")
        uids12 = data12.get("recent_uids")
        if r12_del.status_code == 204 and r12_get.status_code == 200 and cnt12 == 0 and uids12 == []:
            test_results.append((
                "(12) DELETE /session: 204; GET /session sau đó: event_count 0 và recent_uids []",
                "PASS",
                f"DELETE 204, GET event_count=0, recent_uids={uids12}",
            ))
        else:
            test_results.append((
                "(12) DELETE /session: 204; GET /session sau đó: event_count 0 và recent_uids []",
                "FAIL",
                f"DELETE mã {r12_del.status_code}, GET mã {r12_get.status_code}, count={cnt12}, uids={uids12}",
            ))
    except Exception as exc:
        test_results.append(("(12) DELETE /session", "FAIL", str(exc)))

    # 13. GET /session và DELETE /session thiếu header: 400
    try:
        r13_get = client.get("/session")
        r13_del = client.delete("/session")
        if r13_get.status_code == 400 and r13_del.status_code == 400:
            test_results.append((
                "(13) GET /session và DELETE /session thiếu header: 400",
                "PASS",
                "GET 400, DELETE 400",
            ))
        else:
            test_results.append((
                "(13) GET /session và DELETE /session thiếu header: 400",
                "FAIL",
                f"GET: {r13_get.status_code}, DELETE: {r13_del.status_code}",
            ))
    except Exception as exc:
        test_results.append(("(13) Thiếu header /session", "FAIL", str(exc)))

    # 14. journal_mode của data/app.db là 'wal'
    try:
        mode = get_journal_mode(SQLITE_PATH)
        if mode == "wal":
            test_results.append((
                "(14) journal_mode của data/app.db là 'wal'",
                "PASS",
                f"journal_mode={mode}",
            ))
        else:
            test_results.append((
                "(14) journal_mode của data/app.db là 'wal'",
                "FAIL",
                f"journal_mode={mode}",
            ))
    except Exception as exc:
        test_results.append(("(14) journal_mode của data/app.db", "FAIL", str(exc)))

    # In bảng kết quả
    print()
    print("| Tên kiểm tra | Kết quả | Giá trị thực tế |")
    print("| --- | --- | --- |")
    pass_count = 0
    fail_count = 0
    for name, res, actual in test_results:
        if res == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        print(f"| {name} | {res} | {actual} |")

    print()
    print(f"Tổng số: {pass_count} PASS, {fail_count} FAIL")

    return fail_count == 0


if __name__ == "__main__":
    success = run_smoke_tests()
    if not success:
        sys.exit(1)
