"""Cấu hình tải trước (preload) các tệp chỉ mục Lucene vào bộ nhớ đệm trang của hệ điều hành theo SPEC.md."""

import sys
import time
from typing import List

from config import INDEX_NAME
from retrieval.client import get_document_count, get_opensearch_client

# Danh sách phần mở rộng tệp Lucene cần preload vào OS page cache:
# - doc, pos, pay, tim, tip: chỉ mục đảo, từ điển thuật ngữ và vị trí token cho BM25
# - dvd, dvm, nvd, nvm: doc values và trường độ dài văn bản (norms) cho chấm điểm BM25
# - cfs, cfe: tệp compound đóng gói các segment
# - vec, vem, vex, vemf, hnsw: tệp vector float và đồ thị k-NN HNSW của Lucene / OpenSearch
PRELOAD_EXTENSIONS: List[str] = [
    "doc",
    "pos",
    "pay",
    "tim",
    "tip",
    "dvd",
    "dvm",
    "nvd",
    "nvm",
    "cfs",
    "cfe",
    "vec",
    "vem",
    "vex",
    "vemf",
    "hnsw",
]


def apply_store_preload(index_name: str = INDEX_NAME) -> int:
    """Áp dụng cấu hình index.store.preload cho chỉ mục OpenSearch, mở lại và chờ trạng thái green."""
    client = get_opensearch_client()

    print(f"Đóng chỉ mục '{index_name}' để thay đổi thiết lập lưu trữ...")
    client.indices.close(index=index_name, ignore_unavailable=True)

    print(f"Cập nhật index.store.preload với các đuôi: {PRELOAD_EXTENSIONS}...")
    client.indices.put_settings(
        index=index_name,
        body={
            "index": {
                "store": {
                    "preload": PRELOAD_EXTENSIONS,
                }
            }
        },
    )

    print(f"Mở lại chỉ mục '{index_name}'...")
    client.indices.open(index=index_name)

    print(f"Chờ chỉ mục '{index_name}' đạt trạng thái green...")
    client.cluster.health(index=index_name, wait_for_status="green", timeout=60)

    doc_count = get_document_count(index_name)
    print(f"Áp dụng preload thành công cho '{index_name}'. Tổng số tài liệu: {doc_count}")
    return doc_count


def main() -> None:
    """Điểm thực thi dòng lệnh của tập lệnh áp dụng preload."""
    try:
        count = apply_store_preload(INDEX_NAME)
        if count == 0:
            print("Cảnh báo: Chỉ mục không có tài liệu nào.", file=sys.stderr)
    except Exception as exc:
        print(f"Lỗi khi cấu hình preload: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
