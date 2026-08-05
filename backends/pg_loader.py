"""The demo runs on committed flat files; this loads the same snapshot into
Postgres+pgvector to show the at-scale path — approximate cosine search plus
plain SQL metadata filters in one statement, instead of an in-process scan."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DDL = """\
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS stories (
    id text PRIMARY KEY,
    title text,
    url text,
    date date,
    points int,
    tier int,
    domain text,
    topic int,
    embedding vector(384)
);
CREATE TABLE IF NOT EXISTS topics (
    id int PRIMARY KEY,
    label text,
    area text,
    size int
);
"""

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS stories_embedding_hnsw "
    "ON stories USING hnsw (embedding vector_cosine_ops)"
)

STORY_INSERT = (
    "INSERT INTO stories (id, title, url, date, points, tier, domain, topic, embedding) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
)

TOPIC_INSERT = "INSERT INTO topics (id, label, area, size) VALUES (%s, %s, %s, %s)"

HYBRID_QUERY = """\
SELECT s.id, s.title, s.date, s.points, 1 - (s.embedding <=> ref.embedding) AS cosine
FROM stories s, (SELECT embedding FROM stories WHERE id = %(like)s) ref
WHERE s.id <> %(like)s AND s.tier <= %(tier_max)s AND s.date >= %(since)s
ORDER BY s.embedding <=> ref.embedding
LIMIT %(limit)s
"""


def vector_literal(vec: np.ndarray | list[float]) -> str:
    """float() first keeps numpy scalar reprs out of the literal, where they
    would break the server-side parse."""
    return "[" + ",".join(f"{float(v):g}" for v in vec) + "]"  # text form: no pgvector client dep


def story_rows(news_df: pd.DataFrame, ids: np.ndarray, vectors: np.ndarray) -> list[tuple]:
    """Tuple order mirrors the stories DDL so the positional insert cannot
    misalign; a story absent from the npz stops the load instead of leaving a
    NULL embedding that would silently vanish from every similarity result."""
    widened = np.asarray(vectors, dtype=np.float32)  # the committed vectors are half precision
    by_id = dict(zip(ids, widened))
    rows = []
    for rec in news_df.itertuples(index=False):
        if rec.id not in by_id:
            raise SystemExit(
                f"{rec.id} is in the csv but has no vector in data/vectors.npz — "
                "regenerate the embeddings with `make data-cached SAVE_VECTORS=1`."
            )
        rows.append(
            (
                rec.id,
                rec.title,
                rec.url if isinstance(rec.url, str) else None,  # NaN urls load as SQL NULL
                rec.date,
                int(rec.points),
                int(rec.tier),
                rec.domain,
                int(rec.topic),
                vector_literal(by_id[rec.id]),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn", default="postgresql://postgres:trendwatch@localhost:5433/trendwatch"
    )
    parser.add_argument("--like", default=None, help="reference story id (default: first csv row)")
    parser.add_argument("--tier-max", type=int, default=2, help="source quality ceiling")
    parser.add_argument("--since", default="2026-01-01", help="drop stories older than this")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    # Deferred import: the driver arrives with `make install-backends`; the app
    # and base CI never install it.
    import psycopg

    news = pd.read_csv(DATA / "news_topics.csv")
    topics = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))
    npz_path = DATA / "vectors.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"{npz_path} is missing — embeddings are not part of the committed "
            "snapshot; run `make data-cached SAVE_VECTORS=1` to produce it."
        )
    npz = np.load(npz_path)
    rows = story_rows(news, npz["ids"], npz["vectors"])
    topic_rows = [(t["id"], t["label"], t["area"], t["size"]) for t in topics]
    like = args.like or news["id"].iloc[0]
    if like not in set(news["id"]):
        raise SystemExit(f"--like {like}: no such story id in data/news_topics.csv")

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        for statement in DDL.split(";"):
            if statement.strip():
                conn.execute(statement)  # the extended protocol takes one statement per call
        conn.execute("DELETE FROM stories")
        conn.execute("DELETE FROM topics")  # every load is a fresh copy of the snapshot
        with conn.cursor() as cur:
            cur.executemany(TOPIC_INSERT, topic_rows)
            for start in range(0, len(rows), 1000):  # keeps each round trip bounded
                cur.executemany(STORY_INSERT, rows[start : start + 1000])
        conn.execute(INDEX_DDL)  # built once, over the loaded rows
        params = {"like": like, "tier_max": args.tier_max, "since": args.since, "limit": args.limit}
        hits = conn.execute(HYBRID_QUERY, params).fetchall()

    print(f"loaded {len(rows)} stories and {len(topic_rows)} topics; more like {like}:")
    print(f"{'id':<12} {'date':<11} {'points':>6} {'cosine':>7}  title")
    for sid, title, day, points, cosine in hits:
        print(f"{sid:<12} {day!s:<11} {points:>6} {cosine:7.3f}  {title[:58]}")


if __name__ == "__main__":
    main()
