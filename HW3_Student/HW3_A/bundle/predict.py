#!/usr/bin/env python
"""predict.py — Self-contained embedding inference.

MUST implement exactly 4 functions:
    load_bundle()    -> (model, tokenizer)
    embed(texts)     -> np.ndarray shape (N, 384)
    similarity(a, b) -> float
    info()           -> dict

DO NOT import sentence_transformers. Use raw transformers only.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"

# Public constant expected by tests.
BUNDLE_DIR = str(DEFAULT_MODEL_DIR)

MAX_SEQ_LEN = 256

# Public constant expected by tests.
EMBEDDING_DIM = 384

_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LOADED_FROM = None


def _resolve_model_dir(bundle_dir: str | None = None) -> Path:
    """Resolve the model directory safely.

    Important:
    - If bundle_dir is passed directly, use it.
    - Else, if env var BUNDLE_DIR exists, use it.
    - Else, use the default bundle/model directory.

    We do NOT permanently overwrite the public BUNDLE_DIR constant from env,
    because tests may temporarily set BUNDLE_DIR to an invalid path.
    """
    if bundle_dir is not None:
        return Path(bundle_dir).expanduser().resolve()

    env_bundle_dir = os.getenv("BUNDLE_DIR")
    if env_bundle_dir:
        return Path(env_bundle_dir).expanduser().resolve()

    return Path(BUNDLE_DIR).expanduser().resolve()


def load_bundle(bundle_dir: str | None = None) -> Tuple:
    """Load model and tokenizer from the frozen bundle model directory."""
    global _MODEL, _TOKENIZER, _DEVICE, _LOADED_FROM

    model_dir = _resolve_model_dir(bundle_dir)

    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
        "model.safetensors",
    ]

    missing = [fname for fname in required_files if not (model_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Invalid bundle model directory: {model_dir}. "
            f"Missing files: {missing}"
        )

    if (
        _MODEL is not None
        and _TOKENIZER is not None
        and _LOADED_FROM == str(model_dir)
    ):
        return _MODEL, _TOKENIZER

    torch.manual_seed(0)
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model.to(_DEVICE)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _LOADED_FROM = str(model_dir)

    return _MODEL, _TOKENIZER


def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into L2-normalized float32 vectors."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    if len(texts) == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    cleaned_texts = [
        text if isinstance(text, str) and text.strip() else " "
        for text in texts
    ]

    model, tokenizer = load_bundle()
    device = _DEVICE if _DEVICE is not None else torch.device("cpu")

    encoded = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()

        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        normalized = F.normalize(pooled, p=2, dim=1)

    return normalized.detach().cpu().numpy().astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def info() -> dict:
    """Return metadata about the frozen encoder bundle."""
    model_dir = _resolve_model_dir()
    metadata_path = model_dir.parent / "metadata.json"

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "model_name": metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "model_revision": metadata.get("model_revision"),
        "embedding_dim": EMBEDDING_DIM,
        "max_seq_len": MAX_SEQ_LEN,
        "framework": "transformers",
        "device": str(_DEVICE) if _DEVICE is not None else "not_loaded",
        "bundle_dir": str(model_dir),
    }


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Bundle embed CLI")
    p.add_argument("--text", action="append", default=[], help="repeatable text input")
    p.add_argument("--texts-file", help="JSON list of strings")
    p.add_argument("--out", help="optional .npy output path")
    p.add_argument("--info", action="store_true", help="print info and exit")
    args = p.parse_args()

    if args.info:
        print(json.dumps(info(), indent=2, default=str))
        raise SystemExit(0)

    texts = list(args.text)

    if args.texts_file:
        with open(args.texts_file, encoding="utf-8") as f:
            texts.extend(json.load(f))

    if not texts:
        print("ERROR: provide --text or --texts-file", file=sys.stderr)
        raise SystemExit(2)

    emb = embed(texts)

    if args.out:
        np.save(args.out, emb)
        print(f"Saved {emb.shape} to {args.out}")
    else:
        print(json.dumps([[round(float(x), 6) for x in row] for row in emb]))
