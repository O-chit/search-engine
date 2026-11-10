"""Mô-đun tạo embedding cho sản phẩm theo SPEC.md mục 7."""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from config import (
    EMBED_BATCH_SIZE,
    EMBED_CHECKPOINT_EVERY,
    EMBED_DESC_WORDS,
    PROCESSED_DATA_DIR,
    PRODUCTS_PARQUET_PATH,
)
from indexing.model import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_product_text(row: pd.Series) -> str:
    """Tạo văn bản sản phẩm từ title, brand, bullets và 150 từ đầu của description."""
    title = str(row["title"]) if pd.notna(row.get("title")) else ""
    brand = str(row["brand"]) if pd.notna(row.get("brand")) else ""
    bullets = str(row["bullets"]) if pd.notna(row.get("bullets")) else ""
    desc = str(row["description"]) if pd.notna(row.get("description")) else ""
    desc_words = desc.split()[:EMBED_DESC_WORDS]
    desc_part = " ".join(desc_words)
    return f"{title}. {brand}. {bullets}. {desc_part}"


def run_embed(
    parquet_path: str = PRODUCTS_PARQUET_PATH,
    out_dir: str = PROCESSED_DATA_DIR,
    limit: Optional[int] = None,
    checkpoint_every: int = EMBED_CHECKPOINT_EVERY,
    stop_after: Optional[int] = None,
    initial_batch_size: int = EMBED_BATCH_SIZE,
) -> None:
    """Sinh vector embedding cho sản phẩm với hỗ trợ checkpoint và giảm nửa batch khi OOM."""
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Đang đọc dữ liệu từ %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.iloc[:limit]

    total_targets = len(df)
    uids = df["product_uid"].astype(str).tolist()
    texts = [build_product_text(row) for _, row in df.iterrows()]

    ckpt_embs_path = os.path.join(out_dir, "checkpoint_embeddings.npy")
    ckpt_uids_path = os.path.join(out_dir, "checkpoint_uids.json")

    resumed_embeddings: Optional[np.ndarray] = None
    start_idx = 0

    if os.path.exists(ckpt_embs_path) and os.path.exists(ckpt_uids_path):
        try:
            with open(ckpt_uids_path, "r", encoding="utf-8") as f:
                ckpt_uids = json.load(f)
            loaded_embs = np.load(ckpt_embs_path)
            if len(ckpt_uids) == len(loaded_embs) and len(ckpt_uids) > 0:
                start_idx = len(ckpt_uids)
                resumed_embeddings = loaded_embs
                print(f"resume from {start_idx}")
                logger.info("Tiếp tục từ checkpoint: resume from %d", start_idx)
        except Exception as e:
            logger.warning("Không thể đọc checkpoint dở (%s), sẽ chạy từ đầu.", e)
            resumed_embeddings = None
            start_idx = 0

    model = get_embedding_model()
    batch_size = initial_batch_size
    all_chunks: List[np.ndarray] = []
    if resumed_embeddings is not None:
        all_chunks.append(resumed_embeddings)

    current_idx = start_idx
    target_count = min(total_targets, stop_after) if stop_after is not None else total_targets

    while current_idx < target_count:
        next_checkpoint = (
            ((current_idx // checkpoint_every) + 1) * checkpoint_every
            if checkpoint_every > 0
            else target_count
        )
        chunk_end = min(current_idx + batch_size, target_count, next_checkpoint)
        batch_texts = texts[current_idx:chunk_end]

        try:
            batch_vecs = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batch_vecs = batch_vecs.astype(np.float32)
            all_chunks.append(batch_vecs)
            current_idx = chunk_end

            print(f"Tiến độ: {current_idx}/{total_targets} sản phẩm đã xử lý (batch_size={batch_size}).")

            # Lưu checkpoint nếu chạm mốc checkpoint_every
            if current_idx % checkpoint_every == 0 and current_idx < total_targets:
                merged_so_far = np.concatenate(all_chunks, axis=0)
                np.save(ckpt_embs_path, merged_so_far)
                with open(ckpt_uids_path, "w", encoding="utf-8") as f:
                    json.dump(uids[:current_idx], f)
                print(f"Đã lưu checkpoint: {current_idx} sản phẩm.")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            err_msg = str(exc).lower()
            if "out of memory" in err_msg or isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning("Hết bộ nhớ GPU! Giảm batch size xuống %d và thử lại.", batch_size)
                    continue
                else:
                    raise
            else:
                raise

        if stop_after is not None and current_idx >= stop_after:
            merged_so_far = np.concatenate(all_chunks, axis=0)
            np.save(ckpt_embs_path, merged_so_far)
            with open(ckpt_uids_path, "w", encoding="utf-8") as f:
                json.dump(uids[:current_idx], f)
            print(f"Đã lưu checkpoint tại {current_idx} sản phẩm theo --stop-after {stop_after}.")
            break

    total_time = time.perf_counter() - start_time
    processed_count = current_idx - start_idx
    speed = processed_count / total_time if total_time > 0 else 0.0

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    if current_idx >= total_targets:
        final_embeddings = np.concatenate(all_chunks, axis=0)
        final_embs_path = os.path.join(out_dir, "embeddings.npy")
        final_uids_path = os.path.join(out_dir, "uid_index.json")

        np.save(final_embs_path, final_embeddings)
        uid_index = {uid: idx for idx, uid in enumerate(uids[: len(final_embeddings)])}
        with open(final_uids_path, "w", encoding="utf-8") as f:
            json.dump(uid_index, f, indent=2)

        if os.path.exists(ckpt_embs_path):
            os.remove(ckpt_embs_path)
        if os.path.exists(ckpt_uids_path):
            os.remove(ckpt_uids_path)

    print(
        f"Hoàn thành: số sản phẩm = {current_idx}, "
        f"thời gian = {total_time:.2f}s, "
        f"tốc độ = {speed:.2f} sản phẩm/giây, "
        f"batch size cuối cùng = {batch_size}, "
        f"bộ nhớ GPU đỉnh = {peak_vram_mb:.2f} MB"
    )

    if stop_after is not None and current_idx >= stop_after:
        sys.exit(0)


def main() -> None:
    """Điểm vào chính chạy embed từ dòng lệnh."""
    parser = argparse.ArgumentParser(description="Sinh vector embedding cho sản phẩm.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ lấy N sản phẩm đầu")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=EMBED_CHECKPOINT_EVERY,
        help="Lưu checkpoint sau mỗi K sản phẩm",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Dừng có chủ đích sau M sản phẩm và lưu checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="Thư mục ghi kết quả embeddings.npy và uid_index.json",
    )
    args = parser.parse_args()

    run_embed(
        out_dir=args.out_dir,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
