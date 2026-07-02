"""app.client_pg — Postgres read-only client (audit + source of truth)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

import psycopg
from psycopg.rows import dict_row

from . import config


@contextmanager
def _connect() -> Iterator[psycopg.Connection]:
    """Open a short-lived Postgres connection and always close it."""
    conn = None

    try:
        conn = psycopg.connect(
            config.DATABASE_URL,
            connect_timeout=5,
            row_factory=dict_row,
        )
        yield conn

    finally:
        if conn is not None:
            conn.close()


def ping() -> bool:
    """Return True if Postgres is reachable, False otherwise."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

        return True

    except Exception:
        return False


def fetch_corpus_hits(ids: List[str]) -> List[dict]:
    """Fetch rows from core.encoder_corpus in the same order as ids.

    Args:
        ids: UUID strings from Qdrant point IDs.

    Returns:
        List of dict rows:
          id, text, primary_label, labels, lang, source

        The returned order matches the incoming ids order.
    """
    if not ids:
        return []

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id::text AS id,
                    text,
                    primary_label,
                    labels,
                    lang,
                    source
                FROM core.encoder_corpus
                WHERE id = ANY(%s::uuid[])
                """,
                (ids,),
            )

            rows = cur.fetchall()

    by_id = {row["id"]: dict(row) for row in rows}

    return [
        by_id[item_id]
        for item_id in ids
        if item_id in by_id
    ]


def count_corpus() -> Optional[int]:
    """Return corpus row count, or None when Postgres is unavailable."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM core.encoder_corpus")
                row = cur.fetchone()

        if row is None:
            return None

        return int(row["n"])

    except Exception:
        return None