"""Cấu hình toàn cục cho hệ thống tìm kiếm theo SPEC.md mục 13."""

import os
from dotenv import load_dotenv

load_dotenv()


def resolve_device(requested: str = "cuda") -> str:
    """Xác định thiết bị tính toán (cuda hoặc cpu) theo môi trường phần cứng."""
    if requested.lower() == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return requested


# OpenSearch
OPENSEARCH_URL: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME: str = os.getenv("INDEX_NAME", "products")

# Embedding
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1024"))
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))
DEVICE: str = resolve_device(os.getenv("DEVICE", "cuda"))

# Retrieval
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "200"))
CANDIDATE_K: int = int(os.getenv("CANDIDATE_K", "100"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
W_BM25_EN: float = float(os.getenv("W_BM25_EN", "1.0"))
W_VEC_EN: float = float(os.getenv("W_VEC_EN", "1.0"))
W_BM25_VI: float = float(os.getenv("W_BM25_VI", "0.3"))
W_VEC_VI: float = float(os.getenv("W_VEC_VI", "1.0"))
BRANCH_TIMEOUT_MS: int = int(os.getenv("BRANCH_TIMEOUT_MS", "150"))
KNN_EF_SEARCH: int = int(os.getenv("KNN_EF_SEARCH", "100"))

# Ranking
RANKER: str = os.getenv("RANKER", "rrf")
ENABLE_CROSS_ENCODER: bool = os.getenv("ENABLE_CROSS_ENCODER", "false").lower() in ("true", "1", "yes")
CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
CROSS_ENCODER_TOP_N: int = int(os.getenv("CROSS_ENCODER_TOP_N", "20"))

# Personalization
ETA: float = float(os.getenv("ETA", "0.2"))
WARMUP_EVENTS: int = int(os.getenv("WARMUP_EVENTS", "5"))
BETA_PROFILE: float = float(os.getenv("BETA_PROFILE", "0.15"))
BETA_RECENT: float = float(os.getenv("BETA_RECENT", "0.25"))
RECENT_MAX: int = int(os.getenv("RECENT_MAX", "5"))
EXPLORE_POSITION: int = int(os.getenv("EXPLORE_POSITION", "8"))

# Storage
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "data/app.db")

# Ingestion
RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
PRODUCTS_PARQUET_PATH: str = os.path.join(PROCESSED_DATA_DIR, "products.parquet")
ATTR_MAX: int = int(os.getenv("ATTR_MAX", "30"))
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "200"))

