"""Quản lý và nạp mô hình embedding dùng chung theo SPEC.md mục 7."""

import json
import os
from typing import Any, Dict, Optional
import numpy as np

from config import (
    DEVICE,
    EMBED_MAX_SEQ_LEN,
    EMBED_MODEL,
    EMBEDDINGS_NPY_PATH,
    HF_HOME,
    UID_INDEX_PATH,
)

_model: Optional[Any] = None
_embeddings: Optional[Any] = None
_uid_index: Optional[Dict[str, int]] = None


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


def get_product_embeddings() -> Optional[Any]:
    """Nạp lười mảng embeddings.npy dạng mmap dùng chung cho toàn hệ thống."""
    global _embeddings
    if _embeddings is None and os.path.exists(EMBEDDINGS_NPY_PATH):
        _embeddings = np.load(EMBEDDINGS_NPY_PATH, mmap_mode="r")
    return _embeddings


def get_uid_index() -> Optional[Dict[str, int]]:
    """Nạp lười chỉ mục uid_index.json ánh xạ product_uid sang chỉ số vector."""
    global _uid_index
    if _uid_index is None and os.path.exists(UID_INDEX_PATH):
        with open(UID_INDEX_PATH, "r", encoding="utf-8") as f:
            _uid_index = json.load(f)
    return _uid_index

