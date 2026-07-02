"""app.search — hybrid (Qdrant ANN + PG audit) search orchestration."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from . import client_pg, client_qdrant
from . import config
from .schemas import SearchHit


def _payload_to_row(point) -> dict:
    """Fallback row builder using Qdrant payload when Postgres is unavailable."""
    payload = getattr(point, "payload", None) or {}

    primary = (
        payload.get("primary_label")
        or payload.get("primary")
        or payload.get("label")
        or "unknown"
    )

    labels = payload.get("labels", [])

    if labels is None:
        labels = []

    if isinstance(labels, str):
        labels = [labels]

    return {
        "id": str(getattr(point, "id")),
        "text": payload.get("text", ""),
        "primary_label": primary,
        "labels": list(labels),
        "lang": payload.get("lang", "en"),
        "source": payload.get("source", "qdrant"),
    }


def _score(point) -> float:
    """Extract score from a Qdrant scored point."""
    value = getattr(point, "score", 0.0)

    try:
        return float(value)
    except Exception:
        return 0.0


def hybrid_search(
    query_vector: Sequence[float],
    top_k: int,
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = True,
) -> tuple[List[SearchHit], float]:
    """Run Qdrant ANN search, then enrich results from Postgres.

    Returns:
        (hits, took_ms)
    """
    t0 = time.perf_counter()

    qdr_hits = client_qdrant.search(
        collection=config.QDRANT_COLLECTION,
        vector=query_vector,
        top_k=top_k,
        lang=lang,
        primary=primary,
        exclude_neutral=exclude_neutral,
    )

    if not qdr_hits:
        took_ms = (time.perf_counter() - t0) * 1000.0
        return [], took_ms

    ids = [str(hit.id) for hit in qdr_hits]

    try:
        pg_rows = client_pg.fetch_corpus_hits(ids)
    except Exception:
        pg_rows = []

    rows_by_id = {
        str(row["id"]): row
        for row in pg_rows
        if row and row.get("id") is not None
    }

    hits: List[SearchHit] = []

    for point in qdr_hits:
        point_id = str(point.id)
        row = rows_by_id.get(point_id)

        if row is None:
            row = _payload_to_row(point)

        labels = row.get("labels") or []

        if isinstance(labels, str):
            labels = [labels]

        hit = SearchHit(
            id=point_id,
            score=_score(point),
            text=row.get("text", ""),
            primary=row.get("primary_label") or row.get("primary") or "unknown",
            labels=list(labels),
            lang=row.get("lang", "en"),
            source=row.get("source", "unknown"),
        )

        hits.append(hit)

    took_ms = (time.perf_counter() - t0) * 1000.0

    return hits, took_ms