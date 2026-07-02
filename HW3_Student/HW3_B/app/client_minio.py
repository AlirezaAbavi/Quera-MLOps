"""app.client_minio — optional MinIO helper for bundle availability checks.

The API should not depend on MinIO during normal serving when MODEL_SOURCE=baked.
This module is intentionally defensive: failures return False/None instead of
crashing /health.
"""
from __future__ import annotations

from typing import Optional

from minio import Minio

from . import config


_client: Optional[Minio] = None


def _normalize_endpoint(endpoint: str) -> tuple[str, bool]:
    """Return endpoint without scheme and secure flag."""
    endpoint = endpoint.strip()

    if endpoint.startswith("https://"):
        return endpoint.removeprefix("https://"), True

    if endpoint.startswith("http://"):
        return endpoint.removeprefix("http://"), False

    return endpoint, False


def get_client() -> Optional[Minio]:
    """Create one lazy MinIO client.

    Returns None when credentials are not configured.
    """
    global _client

    if _client is not None:
        return _client

    if not config.MINIO_ENDPOINT:
        return None

    if not config.MINIO_ACCESS_KEY or not config.MINIO_SECRET_KEY:
        return None

    endpoint, secure = _normalize_endpoint(config.MINIO_ENDPOINT)

    _client = Minio(
        endpoint=endpoint,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=secure,
    )

    return _client


def ping() -> bool:
    """Return True if MinIO is reachable."""
    try:
        client = get_client()

        if client is None:
            return False

        client.bucket_exists(config.MINIO_BUCKET)
        return True

    except Exception:
        return False


def bundle_exists() -> bool:
    """Return True if the configured bundle prefix appears to exist in MinIO."""
    try:
        client = get_client()

        if client is None:
            return False

        prefix = config.MINIO_PREFIX.strip("/")

        if prefix:
            prefix = prefix + "/"

        objects = client.list_objects(
            bucket_name=config.MINIO_BUCKET,
            prefix=prefix,
            recursive=True,
        )

        for _ in objects:
            return True

        return False

    except Exception:
        return False