"""Quản lý và nạp mô hình embedding dùng chung theo SPEC.md mục 7."""

import os
from typing import Any, Optional

from config import DEVICE, EMBED_MAX_SEQ_LEN, EMBED_MODEL, HF_HOME

_model: Optional[Any] = None


def get_embedding_model() -> Any:
    """Nạp lười mô hình BAAI/bge-m3, chuyển sang fp16 nếu dùng CUDA và đặt max_seq_length."""
    global _model
    if _model is None:
        import transformers.modeling_utils
        import transformers.utils.import_utils

        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            EMBED_MODEL,
            cache_folder=os.path.abspath(HF_HOME),
            device=DEVICE,
            model_kwargs={"weights_only": False},
        )
        model.max_seq_length = EMBED_MAX_SEQ_LEN
        if DEVICE == "cuda":
            model.half()
        _model = model
    return _model
