"""app.predictor — thin wrapper over the bundle's BundlePredictor.

This module exists so the FastAPI layer doesn't import from `predict` directly
(safer refactoring — if the bundle's module name ever changes, only this file
needs to change).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from app import config


def embed_texts(predictor, texts: Sequence[str]) -> np.ndarray:
    """Embed texts using the loaded HW3_A bundle predictor.

    Returns:
        np.ndarray with shape (B, 384), dtype float32, L2-normalized.
    """
    texts = list(texts)

    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    arr = predictor.embed(texts)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={arr.shape}")

    if arr.shape[0] != len(texts):
        raise ValueError(
            f"Embedding batch size mismatch: expected {len(texts)}, got {arr.shape[0]}"
        )

    if arr.shape[1] != config.EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {config.EMBED_DIM}, got {arr.shape[1]}"
        )

    return arr.astype(np.float32, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix with shape (a_rows, b_rows)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)

    if b.ndim == 1:
        b = b.reshape(1, -1)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {a.shape} and {b.shape}")

    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimension mismatch: {a.shape[1]} vs {b.shape[1]}")

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm = np.where(a_norm == 0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0, 1.0, b_norm)

    a_n = a / a_norm
    b_n = b / b_norm

    return a_n @ b_n.T