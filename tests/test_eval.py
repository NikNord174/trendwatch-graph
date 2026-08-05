"""The eval harness only earns trust if its arithmetic is beyond doubt and the
golden set cannot silently drift away from the committed snapshot."""

from pathlib import Path

import pandas as pd

from eval import run

ROOT = Path(__file__).resolve().parent.parent


def test_order_agreement_anchors_and_degenerate_cases():
    reference = ["a", "b", "c"]
    assert run.order_agreement(["a", "b", "c"], reference) == 1.0
    assert run.order_agreement(["c", "b", "a"], reference) == 0.0
    assert run.order_agreement(["b", "x"], reference) is None  # one shared url: no pairs
    assert run.order_agreement([], reference) is None


def test_order_agreement_scores_partial_disorder_between_the_anchors():
    reference = ["a", "b", "c"]
    assert run.order_agreement(["b", "x", "a", "c"], reference) == 2 / 3
    assert run.order_agreement(["a", "c", "b"], reference) == 2 / 3


def test_order_agreement_ignores_repeated_system_urls():
    reference = ["a", "b", "c"]
    assert run.order_agreement(["a", "a", "b"], reference) == 1.0  # no self-pair from the repeat


def test_hit_at_k_only_counts_the_top_k():
    system = ["u1", "u2", "u3"]
    assert run.hit_at_k(system, ["u3", "u9"], 3)
    assert not run.hit_at_k(system, ["u3"], 2)
    assert not run.hit_at_k(system, ["u9"], 8)
    assert not run.hit_at_k([], ["u1"], 1)


def test_story_url_passthrough_and_item_page_fallback():
    assert run.story_url("s:1", "https://example.com/x") == "https://example.com/x"
    assert run.story_url("s:46828881", "") == "https://news.ycombinator.com/item?id=46828881"
    assert run.story_url("s:7", float("nan")) == "https://news.ycombinator.com/item?id=7"


def test_golden_items_match_schema_and_snapshot():
    """A golden url that is not verbatim in news_topics.csv would zero every
    score without any error, so the whole set is checked against the csv."""
    golden = run.load_golden()
    known = set(pd.read_csv(ROOT / "data" / "news_topics.csv")["url"].dropna())
    assert golden
    assert len({item["id"] for item in golden}) == len(golden)
    for item in golden:
        assert set(item) - {"note"} == {"id", "question", "sources"}  # note is optional colour
        assert item["question"].strip()
        assert 2 <= len(item["sources"]) <= 4
        for url in item["sources"]:
            assert url.startswith("http")
            assert url in known, url
