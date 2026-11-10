"""Gói tìm kiếm sản phẩm và các quy trình xử lý pipeline."""

import os

# Dọn dẹp api/pipeline.py cũ theo yêu cầu dọn cấu trúc
_old_pipeline = os.path.join(os.path.dirname(__file__), "..", "api", "pipeline.py")
if os.path.exists(_old_pipeline):
    try:
        os.remove(_old_pipeline)
    except OSError:
        pass
