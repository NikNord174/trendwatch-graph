"""Integrity checks over the committed snapshot — the data the app ships with."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from pipeline import dedupe

ROOT = Path(__file__).resolve().parent.parent


def load_js(name: str) -> dict:
    raw = (ROOT / "assets" / name).read_text(encoding="utf-8")
    return json.loads(raw.partition("=")[2].strip().rstrip(";"))


@pytest.fixture(scope="module")
def graph() -> dict:
    return load_js("graph_data.js")


@pytest.fixture(scope="module")
def relations() -> dict:
    return load_js("relations_data.js")


def test_graph_links_resolve(graph):
    ids = {n["id"] for n in graph["nodes"]}
    for link in graph["links"]:
        assert link["source"] in ids and link["target"] in ids


def test_graph_positions_finite(graph):
    for n in graph["nodes"]:
        assert math.isfinite(n["x"]) and math.isfinite(n["y"])


def test_graph_story_fields_valid(graph):
    stories = [n for n in graph["nodes"] if n["t"] == "story"]
    assert len(stories) > 5000
    for n in stories:
        assert n["tier"] in (1, 2, 3, 4)
        assert n["imp"] in ("High", "Medium", "Low")
        pd.Timestamp(n["date"])  # raises on a malformed date


def test_relations_topics_have_profiles(relations):
    topics = [n for n in relations["nodes"] if n["t"] == "topic"]
    assert topics
    for n in topics:
        assert n["report"]
        assert len(n["news"]) <= 8
        assert n["news_total"] >= len(n["news"])


def test_relations_areas_cover_hubs(relations):
    hub_names = {n["label"] for n in relations["nodes"] if n["t"] == "concept"}
    area_names = {a["name"] for a in relations["areas"]}
    assert hub_names == area_names


def test_topics_json_schema():
    rows = json.loads((ROOT / "data" / "topics.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["size"] > 0
        assert all(0 < share <= 1 for share in r["anchors"].values())


def test_news_csv_consistent_with_topics():
    news = pd.read_csv(ROOT / "data" / "news_topics.csv")
    topics = json.loads((ROOT / "data" / "topics.json").read_text(encoding="utf-8"))
    assert set(news.columns) == {"id", "topic", "date", "points", "tier", "domain", "title", "url"}
    assert set(news["topic"]) <= {t["id"] for t in topics}
    assert pd.to_datetime(news["date"], errors="coerce").notna().all()


def test_hashes_match_recomputation():
    meta = json.loads((ROOT / "data" / "hashes.json").read_text(encoding="utf-8"))
    news = pd.read_csv(ROOT / "data" / "news_topics.csv").fillna({"url": ""})
    assert meta["stories"] == len(meta["hashes"]) == len(news)
    sample = news.head(25)
    for row in sample.itertuples():
        story_id = row.id.removeprefix("s:")
        assert meta["hashes"][story_id] == dedupe.content_hash(row.title, row.url)
