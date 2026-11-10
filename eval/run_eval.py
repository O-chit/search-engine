"""Script đánh giá hiệu năng tìm kiếm (Recall@100, NDCG@10, MRR, độ trễ) theo cấu hình."""

import argparse
import asyncio
from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from api.pipeline import get_supported_configs, run_pipeline
from config import (
    CANDIDATE_K,
    EVAL_RESULTS_DIR,
    RELEVANT_THRESHOLD,
    RETRIEVAL_K,
    RRF_K,
    SPLIT_PATH,
    TRAIN_CSV_PATH,
)
from eval.metrics import (
    label_from_relevance,
    mrr,
    ndcg_at_10_truncated,
    recall_at_k,
)
from ingestion.load_homedepot import read_csv_robust


def load_ground_truth(csv_path: str = TRAIN_CSV_PATH) -> Dict[str, Dict[str, float]]:
    """Nạp nhãn relevance cho từng truy vấn từ train.csv và lấy trung bình nếu trùng product_uid."""
    df = read_csv_robust(
        csv_path,
        dtype={"search_term": str, "product_uid": str, "relevance": float},
    )
    df["search_term"] = df["search_term"].fillna("").astype(str)
    df["product_uid"] = df["product_uid"].fillna("").astype(str)
    df["relevance"] = df["relevance"].astype(float)

    grouped = df.groupby(["search_term", "product_uid"])["relevance"].mean()

    gt_map: Dict[str, Dict[str, float]] = {}
    for (term, uid), mean_rel in grouped.items():
        if term not in gt_map:
            gt_map[term] = {}
        gt_map[term][str(uid)] = float(mean_rel)
    return gt_map


def print_comparison_table(results_dir: str = EVAL_RESULTS_DIR) -> str:
    """In bảng so sánh kết quả đánh giá của tất cả các file cấu hình trong thư mục kết quả."""
    if not os.path.exists(results_dir):
        msg = f"Thư mục kết quả {results_dir} không tồn tại."
        print(msg)
        return msg

    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not json_files:
        msg = "Chưa có file kết quả đánh giá nào trong thư mục."
        print(msg)
        return msg

    rows: List[Dict[str, Any]] = []
    for jf in sorted(json_files):
        fpath = os.path.join(results_dir, jf)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            config_name = data.get("config", os.path.splitext(jf)[0])
            num_queries = data.get("num_queries_total", 0)
            metrics = data.get("metrics", {})
            rec = metrics.get("recall_at_100", {}).get("mean", 0.0)
            ndcg = metrics.get("ndcg_at_10", {}).get("mean", 0.0)
            m = metrics.get("mrr", {}).get("mean", 0.0)
            latency = data.get("latency_ms", {})
            p50 = latency.get("p50", 0.0)
            p95 = latency.get("p95", 0.0)
            rows.append({
                "config": config_name,
                "queries": num_queries,
                "recall": rec,
                "ndcg": ndcg,
                "mrr": m,
                "p50": p50,
                "p95": p95,
            })
        except Exception as exc:
            print(f"Lỗi khi đọc file kết quả {jf}: {exc}")

    lines: List[str] = [
        "| Config | Queries | Recall@100 | NDCG@10 | MRR | p50 (ms) | p95 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['queries']} | {r['recall']:.4f} | {r['ndcg']:.4f} | {r['mrr']:.4f} | {r['p50']:.1f} | {r['p95']:.1f} |"
        )
    table_str = "\n".join(lines)
    print("\n=== BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ ===")
    print(table_str)
    print("=====================================\n")
    return table_str


async def evaluate_config(
    config_name: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy đánh giá cấu hình tìm kiếm trên tập test và ghi kết quả ra file JSON."""
    supported = get_supported_configs()
    if config_name not in supported:
        raise ValueError(
            f"Cấu hình '{config_name}' không tồn tại. Các cấu hình hợp lệ: {supported}"
        )

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file split tại: {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_queries: List[str] = split_data.get("test", [])
    if limit is not None and limit > 0:
        test_queries = test_queries[:limit]

    print(f"Nạp nhãn từ {TRAIN_CSV_PATH}...")
    ground_truth_map = load_ground_truth(TRAIN_CSV_PATH)
    print(f"Bắt đầu đánh giá cấu hình '{config_name}' trên {len(test_queries)} truy vấn...")

    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    latencies: List[float] = []

    t_eval_start = time.perf_counter()

    for idx, query in enumerate(test_queries, start=1):
        res = await run_pipeline(query, config_name=config_name, personalize=False)
        candidate_uids = [c.product_uid for c in res.candidates]
        latencies.append(res.timing.total)

        gt = ground_truth_map.get(query, {})
        rec = recall_at_k(candidate_uids, gt, k=CANDIDATE_K, threshold=RELEVANT_THRESHOLD)
        m = mrr(candidate_uids, gt, threshold=RELEVANT_THRESHOLD)
        ndcg = ndcg_at_10_truncated(candidate_uids, gt, k=10, candidate_k=CANDIDATE_K)

        if rec is not None:
            recalls.append(rec)
        if m is not None:
            mrrs.append(m)
        if ndcg is not None:
            ndcgs.append(ndcg)

        if idx % 100 == 0 or idx == len(test_queries):
            cur_rec = np.mean(recalls) if recalls else 0.0
            cur_ndcg = np.mean(ndcgs) if ndcgs else 0.0
            cur_mrr = np.mean(mrrs) if mrrs else 0.0
            cur_p50 = np.percentile(latencies, 50) if latencies else 0.0
            print(
                f"[{idx}/{len(test_queries)}] "
                f"Recall@100: {cur_rec:.4f} | "
                f"NDCG@10: {cur_ndcg:.4f} | "
                f"MRR: {cur_mrr:.4f} | "
                f"p50: {cur_p50:.1f} ms"
            )

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    eval_result: Dict[str, Any] = {
        "config": config_name,
        "run_at": datetime.now().isoformat(),
        "num_queries_total": len(test_queries),
        "metrics": {
            "recall_at_100": {
                "mean": round(mean_recall, 4),
                "num_queries": len(recalls),
            },
            "mrr": {
                "mean": round(mean_mrr, 4),
                "num_queries": len(mrrs),
            },
            "ndcg_at_10": {
                "mean": round(mean_ndcg, 4),
                "num_queries": len(ndcgs),
            },
        },
        "latency_ms": {
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "mean": round(mean_lat, 2),
        },
        "config_params": {
            "retrieval_k": RETRIEVAL_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "relevant_threshold": RELEVANT_THRESHOLD,
        },
    }

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(EVAL_RESULTS_DIR, f"{config_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"\nĐã lưu kết quả đánh giá vào: {out_file}")
    total_time = time.perf_counter() - t_eval_start
    print(f"Tổng thời gian đánh giá: {total_time:.2f}s")

    print_comparison_table(EVAL_RESULTS_DIR)
    return eval_result


def main() -> None:
    """Hàm chính xử lý tham số dòng lệnh và chạy đánh giá."""
    parser = argparse.ArgumentParser(description="Chạy đánh giá chất lượng tìm kiếm theo cấu hình.")
    parser.add_argument("--config", type=str, required=True, help="Tên cấu hình tìm kiếm (ví dụ: bm25)")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng truy vấn đánh giá")

    args = parser.parse_args()

    supported = get_supported_configs()
    if args.config not in supported:
        sys.stderr.write(
            f"Lỗi: Cấu hình '{args.config}' không hợp lệ. Các cấu hình hợp lệ: {supported}\n"
        )
        sys.exit(1)

    asyncio.run(evaluate_config(args.config, limit=args.limit))


if __name__ == "__main__":
    main()
