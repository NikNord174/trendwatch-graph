from pathlib import Path

from pipeline import dedupe


def test_content_hash_ignores_whitespace_and_case():
    a = dedupe.content_hash("Big  News\tToday", "https://example.com/X")
    b = dedupe.content_hash("big news today", "HTTPS://EXAMPLE.COM/X")
    assert a == b


def test_content_hash_changes_with_content():
    a = dedupe.content_hash("Big news today", "https://example.com/x")
    b = dedupe.content_hash("Big news tomorrow", "https://example.com/x")
    c = dedupe.content_hash("Big news today", "https://example.com/y")
    assert len({a, b, c}) == 3


def test_diff_hashes_classifies_all_cases():
    previous = {"1": "aa", "2": "bb", "3": "cc"}
    current = {"1": "aa", "2": "b2", "4": "dd"}
    diff = dedupe.diff_hashes(current, previous)
    assert diff == {"new": ["4"], "changed": ["2"], "removed": ["3"]}


def test_dedupe_stories_keeps_highest_scored_repost():
    stories = [
        {
            "id": "1",
            "title": "Same story",
            "url": "https://e.com/a",
            "points": 60,
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": "2",
            "title": "same  STORY",
            "url": "https://e.com/a",
            "points": 90,
            "created_at": "2026-01-02T00:00:00Z",
        },
        {
            "id": "3",
            "title": "Different story",
            "url": "https://e.com/b",
            "points": 50,
            "created_at": "2026-01-03T00:00:00Z",
        },
    ]
    kept = dedupe.dedupe_stories(stories)
    assert [s["id"] for s in kept] == ["2", "3"]


def test_load_previous_hashes_missing_file(tmp_path: Path):
    assert dedupe.load_previous_hashes(tmp_path / "absent.json") == {}
