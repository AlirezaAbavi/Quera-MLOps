"""app.reindex — BONUS CLI (+3 pts): embed a JSON corpus and push to Qdrant.

Usage:
    python -m app.reindex --source corpus.json --collection qbc12_corpus
    python -m app.reindex --input corpus.json --collection qbc12_corpus

The input JSON is a list:
    {"text": "hello", "primary": "joy", "labels": ["joy"]}

Flags:
    --source / --input  input JSON file
    --limit N           maximum number of rows to process
    --dry-run           compute embeddings but don't upsert
    --since TIMESTAMP   skip rows before this ISO timestamp
    --batch-size N      default 64

This is a BONUS task. The main homework does not require it.
The go_emotions index on the server was populated by the instructor's indexer.
This tool lets YOU re-embed any text corpus of your own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp safely, including timestamps ending with Z."""
    value = value.strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value)


def _filter_since(corpus: list[dict], since: str | None) -> list[dict]:
    """Skip rows where row['timestamp'] is older than --since."""
    if not since:
        return corpus

    threshold = _parse_iso_timestamp(since)
    filtered: list[dict] = []

    for row in corpus:
        ts = row.get("timestamp")

        if not ts:
            filtered.append(row)
            continue

        try:
            row_dt = _parse_iso_timestamp(str(ts))
        except Exception:
            filtered.append(row)
            continue

        if row_dt >= threshold:
            filtered.append(row)

    return filtered


def main() -> int:
    p = argparse.ArgumentParser(description="HW3_B reindex CLI")

    p.add_argument(
        "--source",
        "--input",
        dest="input",
        required=True,
        help="JSON file: list of {text, primary, labels, lang, source}",
    )

    p.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "qbc12_corpus"),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of rows to process",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute but don't upsert",
    )

    p.add_argument(
        "--since",
        default=None,
        help="skip rows before this ISO timestamp",
    )

    args = p.parse_args()

    if args.batch_size <= 0:
        print("FAIL: --batch-size must be positive", file=sys.stderr)
        return 2

    if args.limit is not None and args.limit < 0:
        print("FAIL: --limit must be non-negative", file=sys.stderr)
        return 2

    print(
        f"[reindex] input={args.input} "
        f"collection={args.collection} "
        f"batch={args.batch_size} "
        f"limit={args.limit} "
        f"dry_run={args.dry_run} "
        f"since={args.since}"
    )

    # Lazy imports so `python -m app.reindex --help` stays lightweight.
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    from . import config
    from .model_loader import ModelService
    from .predictor import embed_texts

    # Load bundle.
    svc = ModelService()
    svc.load()

    if not svc.state.loaded:
        print(f"FAIL: bundle not loaded: {svc.state.error}", file=sys.stderr)
        return 2

    predictor = svc.require_predictor()

    try:
        bundle_info = predictor.info()
    except Exception:
        bundle_info = {}

    print(f"[reindex] bundle loaded: {bundle_info.get('bundle_dir', svc.state.bundle_dir)}")

    # Load corpus.
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"FAIL: input file not found: {input_path}", file=sys.stderr)
        return 2

    with input_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        print(
            f"FAIL: input must be a JSON list, got {type(corpus).__name__}",
            file=sys.stderr,
        )
        return 2

    corpus = _filter_since(corpus, args.since)

    if args.limit is not None:
        corpus = corpus[:args.limit]

    print(f"[reindex] {len(corpus)} rows loaded from {input_path}")

    # Embed in batches.
    qc = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
        timeout=30.0,
    )

    t0 = time.perf_counter()
    points = []

    for i in range(0, len(corpus), args.batch_size):
        batch = corpus[i:i + args.batch_size]
        texts = [str(row["text"]) for row in batch]

        arr = embed_texts(predictor, texts)

        for j, (vec, row) in enumerate(zip(arr, batch)):
            point_id = row.get("id", i + j)

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "text": row["text"],
                        "primary": row.get("primary", "neutral"),
                        "primary_label": row.get("primary", "neutral"),
                        "labels": row.get("labels", []),
                        "lang": row.get("lang", "en"),
                        "source": row.get("source", "instructor_indexer"),
                    },
                )
            )

    if args.dry_run:
        print(f"[reindex] DRY-RUN — would upsert {len(points)} points to {args.collection}")
        return 0

    # Upsert.
    qc.upsert(
        collection_name=args.collection,
        points=points,
        wait=False,
    )

    elapsed = time.perf_counter() - t0
    rate = len(points) / elapsed if elapsed > 0 else 0.0

    print(f"[reindex] upserted {len(points)} points in {elapsed:.1f}s ({rate:.1f} pts/s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())