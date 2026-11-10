"""Đánh chỉ mục dữ liệu sản phẩm hàng loạt vào OpenSearch."""

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
import opensearchpy.helpers
import pandas as pd

from config import (
    BULK_BATCH_SIZE,
    EMBEDDINGS_NPY_PATH,
    INDEX_NAME,
    MAPPING_PATH,
    PRODUCTS_PARQUET_PATH,
    UID_INDEX_PATH,
)
from retrieval.client import create_index, get_opensearch_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_bulk_actions(
    df: pd.DataFrame,
    index_name: str,
    embeddings: Optional[Any] = None,
    uid_index: Optional[Dict[str, int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Tạo generator các tài liệu cần bulk index vào OpenSearch."""
    records = df.to_dict(orient="records")
    for idx, row in enumerate(records):
        uid_str = str(row["product_uid"])
        source: Dict[str, Any] = {
            "product_uid": uid_str,
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
            "brand": str(row["brand"]) if pd.notna(row["brand"]) else "",
            "bullets": str(row["bullets"]) if pd.notna(row["bullets"]) else "",
            "description": str(row["description"]) if pd.notna(row["description"]) else "",
            "attributes_text": (
                str(row["attributes_text"]) if pd.notna(row["attributes_text"]) else ""
            ),
        }
        if embeddings is not None:
            if uid_index is not None and uid_str in uid_index:
                source["embedding"] = embeddings[uid_index[uid_str]].tolist()
            elif idx < len(embeddings):
                source["embedding"] = embeddings[idx].tolist()

        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": uid_str,
            "_source": source,
        }


def run_bulk_index(
    parquet_path: str = PRODUCTS_PARQUET_PATH,
    index_name: str = INDEX_NAME,
    mapping_path: str = MAPPING_PATH,
    recreate: bool = False,
    embeddings_path: str = EMBEDDINGS_NPY_PATH,
    uid_index_path: str = UID_INDEX_PATH,
) -> int:
    """Thực thi toàn bộ luồng tạo chỉ mục và nạp dữ liệu hàng loạt kèm embedding."""
    logger.info("Khởi tạo / kiểm tra chỉ mục %s...", index_name)
    create_index(index_name=index_name, mapping_path=mapping_path, recreate=recreate)

    logger.info("Đang đọc dữ liệu từ %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    total_docs = len(df)
    logger.info("Tổng số sản phẩm cần nạp: %d", total_docs)

    embeddings: Optional[Any] = None
    uid_index: Optional[Dict[str, int]] = None

    if os.path.exists(embeddings_path) and os.path.exists(uid_index_path):
        logger.info("Đang nạp embeddings từ %s và uid_index từ %s...", embeddings_path, uid_index_path)
        embeddings = np.load(embeddings_path, mmap_mode="r")
        with open(uid_index_path, "r", encoding="utf-8") as f:
            uid_index = json.load(f)
        logger.info("Đã nạp embeddings: shape=%s, uid_index: %d mục", embeddings.shape, len(uid_index))
    else:
        logger.warning("Không tìm thấy file embeddings hoặc uid_index!")

    client = get_opensearch_client()
    actions = generate_bulk_actions(
        df=df,
        index_name=index_name,
        embeddings=embeddings,
        uid_index=uid_index,
    )

    indexed_count = 0
    logger.info("Bắt đầu bulk index với batch size = %d...", BULK_BATCH_SIZE)
    for success, _ in opensearchpy.helpers.streaming_bulk(
        client,
        actions,
        chunk_size=BULK_BATCH_SIZE,
        request_timeout=180,
    ):
        if success:
            indexed_count += 1
            if indexed_count % 10000 == 0 or indexed_count == total_docs:
                logger.info(
                    "Đã nạp: %d/%d tài liệu (%.1f%%)",
                    indexed_count,
                    total_docs,
                    (indexed_count / total_docs) * 100,
                )

    client.indices.refresh(index=index_name)
    logger.info("Đã hoàn tất bulk index. Tổng số tài liệu: %d", indexed_count)
    return indexed_count


def main() -> None:
    """Điểm vào chính của script bulk index từ dòng lệnh."""
    parser = argparse.ArgumentParser(description="Bulk index sản phẩm vào OpenSearch.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xoá và tạo lại chỉ mục nếu đã tồn tại.",
    )
    args = parser.parse_args()
    run_bulk_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
