"""Content hashing and snapshot change detection.

Every story gets a stable SHA-256 over its normalized title and URL. Comparing
the current hash set against the previous snapshot's data/hashes.json tells
which stories are new, changed, or gone since the last pipeline run — the same
mechanism an incremental ingest would use against a database, applied to flat
snapshots here.
"""

import hashlib
import json
import re
from pathlib import Path

WHITESPACE = re.compile(r"\s+")


def content_hash(title: str, url: str) -> str:
    """Hash is insensitive to whitespace and case so cosmetic edits don't count."""
    normalized = WHITESPACE.sub(" ", title).strip().lower() + "\n" + url.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_stories(stories: list[dict]) -> dict[str, str]:
    return {s["id"]: content_hash(s["title"], s["url"]) for s in stories}


def diff_hashes(current: dict[str, str], previous: dict[str, str]) -> dict[str, list[str]]:
    new = sorted(k for k in current if k not in previous)
    changed = sorted(k for k in current if k in previous and current[k] != previous[k])
    removed = sorted(k for k in previous if k not in current)
    return {"new": new, "changed": changed, "removed": removed}


def dedupe_stories(stories: list[dict]) -> list[dict]:
    """Drop reposts: same content hash, keep the highest-scored occurrence."""
    best: dict[str, dict] = {}
    for story in stories:
        key = content_hash(story["title"], story["url"])
        kept = best.get(key)
        if kept is None or story["points"] > kept["points"]:
            best[key] = story
    return sorted(best.values(), key=lambda s: s["created_at"])


def load_previous_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("hashes", {})
