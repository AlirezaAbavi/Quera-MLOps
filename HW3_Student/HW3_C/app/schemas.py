"""app.schemas — Pydantic request/response models.

extra="forbid" everywhere so Swagger rejects typos and unknown fields loudly
instead of silently dropping them (a real production incident class).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Service — health + model-info + root
# ============================================================================

class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "error"]
    bundle_loaded: bool
    bundle_dir: str
    qdrant_reachable: bool
    pg_reachable: bool
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_version: str
    model_id: str
    model_revision: str
    device: str
    max_seq_len: int
    embedding_dim: int
    bundle_dir: str
    qdrant_collection: str
    qdrant_vector_count: Optional[int] = None
    threshold: Optional[float] = None


class RootResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    docs: str
    health: str
    version: str


# ============================================================================
# /embed
# ============================================================================

class EmbedRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "texts": [
                    "I love this so much",
                    "I am so angry right now",
                    "The food was okay I guess",
                ]
            }
        },
    )

    texts: List[str] = Field(..., min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    dim: int
    embeddings: List[List[float]]


# ============================================================================
# /search
# ============================================================================

class SearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "I am so happy right now",
                "top_k": 5,
                "lang": "en",
                "exclude_neutral": True,
            }
        },
    )

    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(10, ge=1, le=100)
    lang: Optional[Literal["en"]] = None
    primary: Optional[str] = Field(None, max_length=64)
    exclude_neutral: bool = Field(True)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    text: str
    primary: str
    labels: List[str]
    lang: str
    source: str


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    count: int
    top_k: int
    took_ms: float
    hits: List[SearchHit]


# ============================================================================
# /predict — single text → predicted emotion label via nearest neighbor
# ============================================================================

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "text": "I love this so much",
            }
        },
    )

    text: str = Field(..., min_length=1, max_length=4096)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    predicted_label: str
    confidence: float
    matched_text: str
    elapsed_ms: float