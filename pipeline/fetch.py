"""Fetch Hacker News stories from the Algolia search API.

Keyless public API. Walks the requested time window newest-first and writes
one JSON object per line to pipeline/raw/stories.jsonl.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

API = "https://hn.algolia.com/api/v1/search_by_date"
PAGE_SIZE = 1000
RAW_DIR = Path(__file__).resolve().parent / "raw"


def fetch_window(since: int, until: int, min_points: int, session: requests.Session) -> list[dict]:
    """Collect all matching stories between two unix timestamps.

    Algolia caps a single query at 1000 hits, so instead of offset paging the
    upper bound shrinks to the oldest timestamp seen so far. Stories sharing
    that exact second could theoretically be skipped; at min_points >= 50 that
    boundary never holds more than a couple of items.
    """
    out: list[dict] = []
    hi = until
    while hi > since:
        params = {
            "tags": "story",
            "hitsPerPage": PAGE_SIZE,
            "numericFilters": f"created_at_i>={since},created_at_i<{hi},points>={min_points}",
        }
        resp = session.get(API, params=params, timeout=30)
        resp.raise_for_status()
        hits = resp.json()["hits"]
        if not hits:
            break
        out.extend(hits)
        if len(hits) < PAGE_SIZE:
            break
        hi = min(h["created_at_i"] for h in hits)
        time.sleep(0.4)
    return out


def clean_url(url: str) -> str:
    """The API occasionally serves mangled URLs (double-pasted, embedded
    markdown); treat those like self-posts rather than shipping dead links."""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")) or " " in url or url.count("http") > 1:
        return ""
    return url


def normalize(hit: dict) -> dict:
    return {
        "id": str(hit["objectID"]),
        "title": (hit.get("title") or "").strip(),
        "url": clean_url(hit.get("url") or ""),
        "points": int(hit.get("points") or 0),
        "comments": int(hit.get("num_comments") or 0),
        "created_at": hit.get("created_at") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180, help="window size, days back from now")
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--out", type=Path, default=RAW_DIR / "stories.jsonl")
    args = parser.parse_args()

    until = int(datetime.now(UTC).timestamp())
    since = until - args.days * 86400
    with requests.Session() as session:
        hits = fetch_window(since, until, args.min_points, session)

    seen: set[str] = set()
    stories = []
    for hit in hits:
        story = normalize(hit)
        if story["id"] in seen or not story["title"]:
            continue
        seen.add(story["id"])
        stories.append(story)
    stories.sort(key=lambda s: s["created_at"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for story in stories:
            f.write(json.dumps(story, ensure_ascii=False) + "\n")
    print(f"{len(stories)} stories -> {args.out}")


if __name__ == "__main__":
    main()
