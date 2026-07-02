"""app.config — environment variable contract for HW3_B.

Mirrors the bundle's contract. If you add an env var here, add it to:
  1. .env.example
  2. entrypoint.sh  (if it's a runtime knob, not a build-time secret)
  3. tests/test_env_contract.py  (if it affects determinism)
"""
from __future__ import annotations

import os

# --- App identity ---
APP_TITLE = "QBC12 HW03-B Encoder Embedding & Search API"
APP_VERSION = "0.1.0"

# --- Bundle location ---
# In container: /app/bundle
# In dev: ../HW3_A/bundle
BUNDLE_DIR = os.getenv("BUNDLE_DIR", "/app/bundle")
BUNDLE_DEVICE = os.getenv("BUNDLE_DEVICE", "cpu")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qbc12-qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qbc12_corpus")

# --- Postgres ---
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "qbc12_hw03_encoder")
DATABASE_API_RO_PASSWORD = os.getenv("DATABASE_API_RO_PASSWORD", "")

_default_database_url = (
    f"postgresql://qbc12_hw03_api_ro:{DATABASE_API_RO_PASSWORD}"
    f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

DATABASE_URL = os.getenv("DATABASE_URL", _default_database_url)

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

STUDENT_USERNAME = os.getenv(
    "STUDENT_USERNAME",
    MLFLOW_TRACKING_USERNAME or "student_unknown",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    f"qbc12_hw03_encoder_{STUDENT_USERNAME}",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    f"qbc12_hw03_encoder_{STUDENT_USERNAME}",
)

# --- MinIO ---
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "baked")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "hw03-bundles")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", f"{STUDENT_USERNAME}/")

# --- Search knobs ---
SEARCH_DEFAULT_TOP_K = 10
SEARCH_MAX_TOP_K = 100
SEARCH_MAX_BATCH_TEXTS = 256

# --- Embedding knobs ---
EMBED_MAX_SEQ_LEN = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", "256"))
EMBED_BATCH_HARD_CAP = 256
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "384"))