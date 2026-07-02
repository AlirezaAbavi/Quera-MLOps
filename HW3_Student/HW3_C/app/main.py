"""app.main — FastAPI entrypoint for HW3_C.

Endpoints:
  GET  /              — service root
  GET  /health        — bundle + Qdrant + PG reachability
  GET  /healthz/live  — liveness/startup probe
  GET  /healthz/ready — readiness probe (200 once the bundle is loaded)
  GET  /version       — running image tag (blue/green check)
  GET  /model-info    — bundle metadata + Qdrant vector count
  POST /embed         — text(s) → 384-dim vectors
  POST /predict       — single text → predicted emotion label
  POST /search        — query → Qdrant ANN + PG audit
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from . import client_pg, client_qdrant, config
from . import predictor as predictor_mod
from .model_loader import ModelService
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
    SearchRequest,
    SearchResponse,
)
from .search import hybrid_search

log = logging.getLogger("hw3_b")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


# Global model service instance.
model_service = ModelService()


def _ensure_model_loaded() -> None:
    """Load bundle lazily if lifespan did not run, then validate state.

    Pytest's TestClient sometimes does not trigger lifespan unless used as a
    context manager. This helper makes endpoints robust in both tests and real
    uvicorn runtime.
    """
    if not model_service.state.loaded and model_service.state.error is None:
        model_service.load()

    if not model_service.state.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_service.state.error or "model not loaded",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model bundle on API startup."""
    log.info("HW3_B starting. BUNDLE_DIR=%s", config.BUNDLE_DIR)

    model_service.load()

    if model_service.state.loaded:
        log.info("Bundle loaded: %s", model_service.state.bundle_dir)
    else:
        log.error("Bundle load FAILED: %s", model_service.state.error)

    yield

    log.info("HW3_B shutting down.")


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", response_model=RootResponse, tags=["service"])
def root() -> RootResponse:
    """Service root."""
    return RootResponse(
        message="QBC12 HW3 Encoder API",
        docs="/docs",
        health="/health",
        version=config.APP_VERSION,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Health probe.

    Important: this endpoint should return 200 even when dependencies are
    degraded. It reports status in the payload instead of throwing 5xx.
    """
    if not model_service.state.loaded and model_service.state.error is None:
        model_service.load()

    bundle_ok = bool(model_service.state.loaded)
    qdrant_ok = client_qdrant.ping()
    pg_ok = client_pg.ping()

    overall = "ok" if bundle_ok and qdrant_ok and pg_ok else "degraded"

    return HealthResponse(
        status=overall,
        bundle_loaded=bundle_ok,
        bundle_dir=str(model_service.state.bundle_dir or config.BUNDLE_DIR),
        qdrant_reachable=qdrant_ok,
        pg_reachable=pg_ok,
        error=model_service.state.error,
    )


# ---------------------------------------------------------------------------
# Probes + version (k8s)
# ---------------------------------------------------------------------------

@app.get("/healthz/live", tags=["k8s"])
def healthz_live() -> dict:
    """Liveness/startup probe. 200 while the process is alive, even mid-load."""
    return {"status": "live"}


@app.get("/healthz/ready", tags=["k8s"])
def healthz_ready(response: Response) -> dict:
    """Readiness probe. 200 only after the bundle is loaded, else 503."""
    if model_service.state.loaded:
        return {"status": "ready", "model_loaded": True}

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_ready", "model_loaded": False}


@app.get("/version", tags=["k8s"])
def version() -> dict:
    """Running image tag, for the blue/green cutover check."""
    return {"image_tag": config.IMAGE_TAG}


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Return loaded bundle metadata and Qdrant collection information."""
    _ensure_model_loaded()

    metadata = model_service.metadata or {}

    qdrant_vector_count = client_qdrant.vector_count(config.QDRANT_COLLECTION)

    return ModelInfoResponse(
        bundle_version=str(metadata.get("model_revision") or config.APP_VERSION),
        model_id=str(metadata.get("model_name") or "unknown"),
        model_revision=str(metadata.get("model_revision") or "unknown"),
        device=str(metadata.get("device") or config.BUNDLE_DEVICE),
        max_seq_len=int(metadata.get("max_seq_len") or config.EMBED_MAX_SEQ_LEN),
        embedding_dim=int(metadata.get("embedding_dim") or config.EMBED_DIM),
        bundle_dir=str(model_service.state.bundle_dir or config.BUNDLE_DIR),
        qdrant_collection=config.QDRANT_COLLECTION,
        qdrant_vector_count=qdrant_vector_count,
        threshold=config.THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed a batch of input texts."""
    _ensure_model_loaded()

    if len(req.texts) > config.EMBED_BATCH_HARD_CAP:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"batch too large; max={config.EMBED_BATCH_HARD_CAP}",
        )

    try:
        vectors = predictor_mod.embed_texts(
            model_service.require_predictor(),
            req.texts,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"embedding failed: {exc}",
        ) from exc

    return EmbedResponse(
        count=len(req.texts),
        dim=int(vectors.shape[1]),
        embeddings=vectors.tolist(),
    )


# ---------------------------------------------------------------------------
# /predict — single text → emotion label via nearest neighbor
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["embedding"])
def predict(req: PredictRequest) -> PredictResponse:
    """Predict a single emotion label using nearest-neighbor search."""
    _ensure_model_loaded()

    try:
        query_vec = predictor_mod.embed_texts(
            model_service.require_predictor(),
            [req.text],
        )

        hits, took_ms = hybrid_search(
            query_vector=query_vec[0].tolist(),
            top_k=1,
            lang=None,
            primary=None,
            exclude_neutral=False,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"prediction backend unavailable: {exc}",
        ) from exc

    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no match found in corpus",
        )

    best = hits[0]

    return PredictResponse(
        text=req.text,
        predicted_label=best.primary,
        confidence=float(best.score),
        matched_text=best.text,
        elapsed_ms=float(took_ms),
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """Embed query text and perform hybrid Qdrant + Postgres search."""
    _ensure_model_loaded()

    try:
        query_vec = predictor_mod.embed_texts(
            model_service.require_predictor(),
            [req.query],
        )

        hits, took_ms = hybrid_search(
            query_vector=query_vec[0].tolist(),
            top_k=req.top_k,
            lang=req.lang,
            primary=req.primary,
            exclude_neutral=req.exclude_neutral,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"search backend unavailable: {exc}",
        ) from exc

    return SearchResponse(
        query=req.query,
        count=len(hits),
        top_k=req.top_k,
        took_ms=float(took_ms),
        hits=hits,
    )