"""Kịch bản kiểm thử khói (smoke test) các tính năng cá nhân hóa theo SPEC.md mục 11."""

import json
import sys
import time
from typing import Any, Dict, List, Tuple
import uuid

import httpx
import numpy as np

from config import API_PORT, EMBED_DIM, SQLITE_PATH
from storage.db import get_events, get_query, get_session


def run_smoke_personalization() -> bool:
    """Thực thi 11 bước kiểm thử khói cá nhân hóa và in bảng kết quả chi tiết."""
    base_url = f"http://127.0.0.1:{API_PORT}"
    session_id = f"smoke_pers_{uuid.uuid4().hex[:12]}"
    test_results: List[Tuple[str, str, str]] = []

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # 1. phiên mới tìm 'tools' (personalize=true): personalized=false và dòng queries.personalized = 0
    try:
        r1 = client.get("/search?q=tools&personalize=true", headers={"X-Session-Id": session_id})
        data1 = r1.json() if r1.status_code == 200 else {}
        qid1 = data1.get("query_id")
        pers1 = data1.get("personalized")
        deg1 = data1.get("degraded", True)
        q_row1 = get_query(qid1) if qid1 else None
        db_pers1 = q_row1.get("personalized") if q_row1 else None

        if r1.status_code == 200 and pers1 is False and db_pers1 == 0 and not deg1:
            test_results.append((
                "(1) phiên mới tìm 'tools' (personalize=true): personalized=false và queries.personalized = 0",
                "PASS",
                f"personalized={pers1}, queries.personalized={db_pers1}, degraded={deg1}",
            ))
        else:
            test_results.append((
                "(1) phiên mới tìm 'tools' (personalize=true): personalized=false và queries.personalized = 0",
                "FAIL",
                f"status={r1.status_code}, personalized={pers1}, db={db_pers1}, degraded={deg1}",
            ))
    except Exception as exc:
        test_results.append(("(1) phiên mới tìm 'tools'", "FAIL", str(exc)))

    # 2. tìm 'paint brush', click lần lượt 6 sản phẩm khác nhau ở vị trí 1..6 (POST /events, mỗi lần chờ ghi xong): GET /session event_count 6, recent_uids có 5 phần tử, phần tử đầu là sản phẩm click cuối
    clicked_uids: List[str] = []
    qid2 = None
    try:
        r2_search = client.get(
            "/search?q=paint%20brush&size=10",
            headers={"X-Session-Id": session_id},
        )
        data2_search = r2_search.json() if r2_search.status_code == 200 else {}
        qid2 = data2_search.get("query_id")
        deg2 = data2_search.get("degraded", True)
        search_results = data2_search.get("results", [])
        target_candidates = search_results[:6]

        for idx, item in enumerate(target_candidates, start=1):
            p_uid = item["product_uid"]
            clicked_uids.append(p_uid)
            r_ev = client.post(
                "/events",
                headers={"X-Session-Id": session_id},
                json={
                    "query_id": qid2,
                    "product_uid": p_uid,
                    "type": "click",
                    "position": idx,
                },
            )
            # Chờ SQLite cập nhật event_count đạt mức idx qua BackgroundTasks
            t_wait = time.perf_counter()
            while time.perf_counter() - t_wait < 3.0:
                s_chk = get_session(session_id)
                if s_chk and int(s_chk.get("event_count") or 0) >= idx:
                    break
                time.sleep(0.05)

        r2_sess = client.get("/session", headers={"X-Session-Id": session_id})
        data2_sess = r2_sess.json() if r2_sess.status_code == 200 else {}
        cnt2 = data2_sess.get("event_count")
        recent2 = data2_sess.get("recent_uids", [])

        is_cnt_ok = cnt2 == 6
        is_len_ok = len(recent2) == 5
        is_first_ok = len(recent2) > 0 and recent2[0] == clicked_uids[-1]

        if r2_sess.status_code == 200 and not deg2 and is_cnt_ok and is_len_ok and is_first_ok:
            test_results.append((
                "(2) tìm 'paint brush', click lần lượt 6 sản phẩm khác nhau: event_count 6, recent_uids 5 phần tử, đầu là click cuối",
                "PASS",
                f"event_count={cnt2}, recent_len={len(recent2)}, first={recent2[0]}, degraded={deg2}",
            ))
        else:
            test_results.append((
                "(2) tìm 'paint brush', click lần lượt 6 sản phẩm khác nhau: event_count 6, recent_uids 5 phần tử, đầu là click cuối",
                "FAIL",
                f"event_count={cnt2}, len={len(recent2)}, first={recent2[0] if recent2 else None}, degraded={deg2}",
            ))
    except Exception as exc:
        test_results.append(("(2) click 6 sản phẩm và kiểm tra /session", "FAIL", str(exc)))

    # 3. dòng sessions có profile_vec dài đúng EMBED_DIM*4 byte, chuẩn L2 trong [0.999, 1.001]
    try:
        s_row3 = get_session(session_id)
        raw_blob = s_row3.get("profile_vec") if s_row3 else None
        if raw_blob and isinstance(raw_blob, (bytes, bytearray)):
            byte_len = len(raw_blob)
            arr = np.frombuffer(raw_blob, dtype=np.float32)
            l2_norm = float(np.linalg.norm(arr))
            if byte_len == EMBED_DIM * 4 and 0.999 <= l2_norm <= 1.001:
                test_results.append((
                    "(3) dòng sessions có profile_vec dài đúng EMBED_DIM*4 byte, chuẩn L2 trong [0.999, 1.001]",
                    "PASS",
                    f"bytes={byte_len}, L2={l2_norm:.6f}",
                ))
            else:
                test_results.append((
                    "(3) dòng sessions có profile_vec dài đúng EMBED_DIM*4 byte, chuẩn L2 trong [0.999, 1.001]",
                    "FAIL",
                    f"bytes={byte_len} (cần {EMBED_DIM*4}), L2={l2_norm:.6f}",
                ))
        else:
            test_results.append((
                "(3) dòng sessions có profile_vec dài đúng EMBED_DIM*4 byte, chuẩn L2 trong [0.999, 1.001]",
                "FAIL",
                f"profile_vec không tồn tại hoặc không phải BLOB bytes: {type(raw_blob)}",
            ))
    except Exception as exc:
        test_results.append(("(3) kiểm tra profile_vec trong SQLite", "FAIL", str(exc)))

    # 4. click lại đúng sản phẩm vừa click trong vòng 1 giây: số dòng events tăng 1 nhưng event_count KHÔNG tăng
    try:
        events_before = len(get_events(session_id))
        sess_before = get_session(session_id)
        cnt_before = int(sess_before.get("event_count") or 0) if sess_before else 0
        last_clicked = clicked_uids[-1]

        r4 = client.post(
            "/events",
            headers={"X-Session-Id": session_id},
            json={
                "query_id": qid2 or "",
                "product_uid": last_clicked,
                "type": "click",
                "position": 6,
            },
        )
        time.sleep(0.5)
        events_after = len(get_events(session_id))
        sess_after = get_session(session_id)
        cnt_after = int(sess_after.get("event_count") or 0) if sess_after else 0

        is_event_inc = events_after == events_before + 1
        is_cnt_same = cnt_after == cnt_before

        if r4.status_code == 202 and is_event_inc and is_cnt_same:
            test_results.append((
                "(4) click lại đúng sản phẩm vừa click trong vòng 1 giây: số dòng events tăng 1 nhưng event_count KHÔNG tăng",
                "PASS",
                f"events: {events_before} -> {events_after}, event_count: {cnt_before} -> {cnt_after}",
            ))
        else:
            test_results.append((
                "(4) click lại đúng sản phẩm vừa click trong vòng 1 giây: số dòng events tăng 1 nhưng event_count KHÔNG tăng",
                "FAIL",
                f"events: {events_before} -> {events_after}, count: {cnt_before} -> {cnt_after}",
            ))
    except Exception as exc:
        test_results.append(("(4) click dedupe trong 1 giây", "FAIL", str(exc)))

    # 5. tìm 'tools' personalize=true: personalized=true, dòng queries.personalized = 1, timing_ms có khoá personalize
    try:
        r5 = client.get("/search?q=tools&personalize=true", headers={"X-Session-Id": session_id})
        data5 = r5.json() if r5.status_code == 200 else {}
        pers5 = data5.get("personalized")
        deg5 = data5.get("degraded", True)
        qid5 = data5.get("query_id")
        timing5 = data5.get("timing_ms", {})
        q_row5 = get_query(qid5) if qid5 else None
        db_pers5 = q_row5.get("personalized") if q_row5 else None

        has_timing_key = "personalize" in timing5 and timing5["personalize"] is not None

        if r5.status_code == 200 and not deg5 and pers5 is True and db_pers5 == 1 and has_timing_key:
            test_results.append((
                "(5) tìm 'tools' personalize=true: personalized=true, queries.personalized = 1, timing_ms có khoá personalize",
                "PASS",
                f"personalized={pers5}, queries.personalized={db_pers5}, timing.personalize={timing5.get('personalize')}, degraded={deg5}",
            ))
        else:
            test_results.append((
                "(5) tìm 'tools' personalize=true: personalized=true, queries.personalized = 1, timing_ms có khoá personalize",
                "FAIL",
                f"status={r5.status_code}, pers={pers5}, db_pers={db_pers5}, timing_keys={list(timing5.keys())}, degraded={deg5}",
            ))
    except Exception as exc:
        test_results.append(("(5) tìm 'tools' personalize=true", "FAIL", str(exc)))

    # 6. cùng truy vấn với debug=true: mọi kết quả có cos_profile và cos_recent là số
    try:
        r6 = client.get("/search?q=tools&personalize=true&debug=true", headers={"X-Session-Id": session_id})
        data6 = r6.json() if r6.status_code == 200 else {}
        deg6 = data6.get("degraded", True)
        res6 = data6.get("results", [])

        valid_numbers = (
            len(res6) > 0
            and all(
                r.get("debug") is not None
                and isinstance(r["debug"].get("cos_profile"), (int, float))
                and isinstance(r["debug"].get("cos_recent"), (int, float))
                for r in res6
            )
        )

        if r6.status_code == 200 and not deg6 and valid_numbers:
            test_results.append((
                "(6) cùng truy vấn với debug=true: mọi kết quả có cos_profile và cos_recent là số",
                "PASS",
                f"Đã kiểm tra {len(res6)} kết quả, mọi cos_profile và cos_recent đều là số thực, degraded={deg6}",
            ))
        else:
            test_results.append((
                "(6) cùng truy vấn với debug=true: mọi kết quả có cos_profile và cos_recent là số",
                "FAIL",
                f"status={r6.status_code}, len={len(res6)}, valid={valid_numbers}, degraded={deg6}",
            ))
    except Exception as exc:
        test_results.append(("(6) debug=true kiểm tra cos_profile/cos_recent", "FAIL", str(exc)))

    # 7. so top 10 của 'tools' personalize=true với personalize=false: in hai danh sách product_uid và số phần tử khác nhau (phải > 0)
    res_tools_p_true = []
    try:
        r7_true = client.get("/search?q=tools&personalize=true&size=10", headers={"X-Session-Id": session_id})
        r7_false = client.get("/search?q=tools&personalize=false&size=10", headers={"X-Session-Id": session_id})
        data7_t = r7_true.json() if r7_true.status_code == 200 else {}
        data7_f = r7_false.json() if r7_false.status_code == 200 else {}
        deg7_t = data7_t.get("degraded", True)
        deg7_f = data7_f.get("degraded", True)
        res_tools_p_true = data7_t.get("results", [])

        uids_true = [r["product_uid"] for r in res_tools_p_true]
        uids_false = [r["product_uid"] for r in data7_f.get("results", [])]

        diff_count = sum(1 for a, b in zip(uids_true, uids_false) if a != b) + abs(len(uids_true) - len(uids_false))

        if r7_true.status_code == 200 and r7_false.status_code == 200 and not deg7_t and not deg7_f and diff_count > 0:
            test_results.append((
                "(7) so top 10 của 'tools' personalize=true với personalize=false: in hai danh sách product_uid và số phần tử khác nhau (> 0)",
                "PASS",
                f"true={uids_true}, false={uids_false}, khác nhau={diff_count} phần tử",
            ))
        else:
            test_results.append((
                "(7) so top 10 của 'tools' personalize=true với personalize=false: in hai danh sách product_uid và số phần tử khác nhau (> 0)",
                "FAIL",
                f"diff_count={diff_count}, deg_t={deg7_t}, deg_f={deg7_f}, true={uids_true}, false={uids_false}",
            ))
    except Exception as exc:
        test_results.append(("(7) so top 10 personalize true vs false", "FAIL", str(exc)))

    # 8. nếu có kết quả explored=true thì nó ở position 8 (in rõ có hay không)
    try:
        explored_positions = [
            (idx + 1, r["product_uid"])
            for idx, r in enumerate(res_tools_p_true)
            if r.get("explored") is True
        ]
        if not explored_positions:
            test_results.append((
                "(8) nếu có kết quả explored=true thì nó ở position 8 (in rõ có hay không)",
                "PASS",
                "Không có ứng viên explored=true",
            ))
        else:
            positions = [p for p, _ in explored_positions]
            if positions == [8]:
                test_results.append((
                    "(8) nếu có kết quả explored=true thì nó ở position 8 (in rõ có hay không)",
                    "PASS",
                    f"Có ứng viên explored=true ở position 8 (product_uid={explored_positions[0][1]})",
                ))
            else:
                test_results.append((
                    "(8) nếu có kết quả explored=true thì nó ở position 8 (in rõ có hay không)",
                    "FAIL",
                    f"explored=true xuất hiện ở vị trí: {positions}",
                ))
    except Exception as exc:
        test_results.append(("(8) kiểm tra vị trí explored", "FAIL", str(exc)))

    # 9. tìm 'dewalt dcd771c2' personalize=true: personalized=false và thứ tự product_uid giống hệt personalize=false
    try:
        r9_t = client.get("/search?q=dewalt%20dcd771c2&personalize=true&size=10", headers={"X-Session-Id": session_id})
        r9_f = client.get("/search?q=dewalt%20dcd771c2&personalize=false&size=10", headers={"X-Session-Id": session_id})
        data9_t = r9_t.json() if r9_t.status_code == 200 else {}
        data9_f = r9_f.json() if r9_f.status_code == 200 else {}
        deg9_t = data9_t.get("degraded", True)
        deg9_f = data9_f.get("degraded", True)

        pers9 = data9_t.get("personalized")
        uids9_t = [r["product_uid"] for r in data9_t.get("results", [])]
        uids9_f = [r["product_uid"] for r in data9_f.get("results", [])]

        is_pers_false = pers9 is False
        is_order_exact = (len(uids9_t) > 0) and (uids9_t == uids9_f)

        if r9_t.status_code == 200 and r9_f.status_code == 200 and not deg9_t and not deg9_f and is_pers_false and is_order_exact:
            test_results.append((
                "(9) tìm 'dewalt dcd771c2' personalize=true: personalized=false và thứ tự product_uid giống hệt personalize=false",
                "PASS",
                f"personalized={pers9}, thứ tự khớp 100% ({len(uids9_t)} sản phẩm), degraded=False",
            ))
        else:
            test_results.append((
                "(9) tìm 'dewalt dcd771c2' personalize=true: personalized=false và thứ tự product_uid giống hệt personalize=false",
                "FAIL",
                f"pers={pers9}, match={is_order_exact}, deg_t={deg9_t}, deg_f={deg9_f}, true={uids9_t}, false={uids9_f}",
            ))
    except Exception as exc:
        test_results.append(("(9) bảo vệ mã model 'dewalt dcd771c2'", "FAIL", str(exc)))

    # 10. DELETE /session rồi tìm 'tools': personalized=false và top 10 giống hệt personalize=false
    try:
        r10_del = client.delete("/session", headers={"X-Session-Id": session_id})
        r10_t = client.get("/search?q=tools&personalize=true&size=10", headers={"X-Session-Id": session_id})
        r10_f = client.get("/search?q=tools&personalize=false&size=10", headers={"X-Session-Id": session_id})
        data10_t = r10_t.json() if r10_t.status_code == 200 else {}
        data10_f = r10_f.json() if r10_f.status_code == 200 else {}
        deg10_t = data10_t.get("degraded", True)
        deg10_f = data10_f.get("degraded", True)

        pers10 = data10_t.get("personalized")
        uids10_t = [r["product_uid"] for r in data10_t.get("results", [])]
        uids10_f = [r["product_uid"] for r in data10_f.get("results", [])]

        is_del_ok = r10_del.status_code == 204
        is_pers_false = pers10 is False
        is_order_exact = uids10_t == uids10_f

        if is_del_ok and not deg10_t and not deg10_f and is_pers_false and is_order_exact:
            test_results.append((
                "(10) DELETE /session rồi tìm 'tools': personalized=false và top 10 giống hệt personalize=false",
                "PASS",
                f"DELETE mã 204, personalized={pers10}, top 10 khớp hoàn toàn với personalize=false, degraded=False",
            ))
        else:
            test_results.append((
                "(10) DELETE /session rồi tìm 'tools': personalized=false và top 10 giống hệt personalize=false",
                "FAIL",
                f"del_code={r10_del.status_code}, pers={pers10}, match={is_order_exact}, deg_t={deg10_t}, deg_f={deg10_f}",
            ))
    except Exception as exc:
        test_results.append(("(10) DELETE /session và tìm 'tools'", "FAIL", str(exc)))

    # 11. tìm 'tools' thiếu header X-Session-Id: 200, personalized=false
    try:
        r11 = client.get("/search?q=tools&personalize=true")
        data11 = r11.json() if r11.status_code == 200 else {}
        pers11 = data11.get("personalized")
        deg11 = data11.get("degraded", True)

        if r11.status_code == 200 and not deg11 and pers11 is False:
            test_results.append((
                "(11) tìm 'tools' thiếu header X-Session-Id: 200, personalized=false",
                "PASS",
                f"Mã 200 OK, personalized={pers11}, degraded={deg11}",
            ))
        else:
            test_results.append((
                "(11) tìm 'tools' thiếu header X-Session-Id: 200, personalized=false",
                "FAIL",
                f"status={r11.status_code}, personalized={pers11}, degraded={deg11}",
            ))
    except Exception as exc:
        test_results.append(("(11) tìm 'tools' thiếu header X-Session-Id", "FAIL", str(exc)))

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
    success = run_smoke_personalization()
    if not success:
        sys.exit(1)
