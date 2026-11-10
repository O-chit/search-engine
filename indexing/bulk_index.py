"""Đánh chỉ mục dữ liệu sản phẩm hàng loạt vào OpenSearch."""

import argparse
import logging
from typing import Any, Dict, Iterator, Optional

import numpy as np
import opensearchpy.helpers
import pandas as pd

from config import (
    BULK_BATCH_SIZE,
    INDEX_NAME,
    MAPPING_PATH,
    PRODUCTS_PARQUET_PATH,
)
from retrieval.client import create_index, get_opensearch_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_bulk_actions(
    df: pd.DataFrame,
    index_name: str,
    embeddings: Optional[np.ndarray] = None,
) -> Iterator[Dict[str, Any]]:
    """Tạo generator các tài liệu cần bulk index vào OpenSearch."""
    for idx, row in df.iterrows():
        doc: Dict[str, Any] = {
            "_op_type": "index",
            "_index": index_name,
            "_id": str(row["product_uid"]),
            "_source": {
                "product_uid": str(row["product_uid"]),
                "title": str(row["title"]) if pd.notna(row["title"]) else "",
                "brand": str(row["brand"]) if pd.notna(row["brand"]) else "",
                "bullets": str(row["bullets"]) if pd.notna(row["bullets"]) else "",
                "description": str(row["description"]) if pd.notna(row["description"]) else "",
                "attributes_text": (
                    str(row["attributes_text"]) if pd.notna(row["attributes_text"]) else ""
                ),
            },
        }
        if embeddings is not None:
            doc["_source"]["embedding"] = embeddings[idx].tolist()
        yield doc


def run_bulk_index(
    parquet_path: str = PRODUCTS_PARQUET_PATH,
    index_name: str = INDEX_NAME,
    mapping_path: str = MAPPING_PATH,
    recreate: bool = False,
    embeddings: Optional[np.ndarray] = None,
) -> int:
    """Thực thi toàn bộ luồng tạo chỉ mục và nạp dữ liệu hàng loạt."""
    logger.info("Khởi tạo / kiểm tra chỉ mục %s...", index_name)
    create_index(index_name=index_name, mapping_path=mapping_path, recreate=recreate)

    logger.info("Đang đọc dữ liệu từ %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    total_docs = len(df)
    logger.info("Tổng số sản phẩm cần nạp: %d", total_docs)

    client = get_opensearch_client()
    actions = generate_bulk_actions(df=df, index_name=index_name, embeddings=embeddings)

    indexed_count = 0
    logger.info("Bắt đầu bulk index với batch size = %d...", BULK_BATCH_SIZE)
    for success, _ in opensearchpy.helpers.streaming_bulk(
        client,
        actions,
        chunk_size=BULK_BATCH_SIZE,
        request_timeout=120,
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
