"""app.client_qdrant — Qdrant client."""

from __future__ import annotations

from typing import List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config


_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Create one lazy singleton Qdrant client."""
    global _client

    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10.0,
        )

    return _client


def ping() -> bool:
    """Return True if Qdrant is reachable, False otherwise."""
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def vector_count(collection: str) -> Optional[int]:
    """Return vector count for a collection, or None on error."""
    try:
        info = get_client().get_collection(collection_name=collection)

        count = getattr(info, "vectors_count", None)
        if count is not None:
            return int(count)

        count = getattr(info, "points_count", None)
        if count is not None:
            return int(count)

        return None

    except Exception:
        return None


def _build_filter(
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = True,
) -> Optional[models.Filter]:
    """Build optional Qdrant payload filter."""
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []

    if lang:
        must.append(
            models.FieldCondition(
                key="lang",
                match=models.MatchValue(value=lang),
            )
        )

    if primary:
        must.append(
            models.FieldCondition(
                key="primary",
                match=models.MatchValue(value=primary),
            )
        )

    if exclude_neutral:
        must_not.append(
            models.FieldCondition(
                key="primary",
                match=models.MatchValue(value="neutral"),
            )
        )

    if not must and not must_not:
        return None

    return models.Filter(
        must=must or None,
        must_not=must_not or None,
    )


def _is_missing_collection_error(exc: Exception, collection: str) -> bool:
    """Detect Qdrant 'collection does not exist' errors."""
    message = str(exc).lower()
    collection_lower = collection.lower()

    return (
        "not found" in message
        and "collection" in message
        and collection_lower in message
    ) or (
        "doesn't exist" in message
        and collection_lower in message
    )


def search(
    collection: str,
    vector: Sequence[float],
    top_k: int,
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = True,
) -> List[models.ScoredPoint]:
    """Run ANN search with optional payload filters.

    If the local Qdrant collection does not exist yet, return [] instead of
    crashing the API. This lets /search degrade gracefully in local Docker.
    """
    query_filter = _build_filter(
        lang=lang,
        primary=primary,
        exclude_neutral=exclude_neutral,
    )

    client = get_client()
    query_vector = list(vector)

    try:
        return client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

    except AttributeError:
        try:
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            points = getattr(result, "points", result)
            return list(points)

        except Exception as exc:
            if _is_missing_collection_error(exc, collection):
                return []

            raise

    except Exception as exc:
        if _is_missing_collection_error(exc, collection):
            return []

        raise