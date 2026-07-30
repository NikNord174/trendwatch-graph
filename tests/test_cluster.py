import numpy as np

from pipeline import cluster


def test_clean_title_strips_hn_prefixes():
    assert cluster.clean_title("Show HN: My side project") == "My side project"
    assert cluster.clean_title("ask hn Why is this") == "Why is this"
    assert cluster.clean_title("Plain title") == "Plain title"


def test_pick_k_bounds():
    assert cluster.pick_k(100) == 40
    assert cluster.pick_k(9_000) == 100
    assert cluster.pick_k(100_000) == 120


def test_drop_redundant_removes_contained_terms():
    terms = ["language models", "models", "rust", "rust compiler"]
    assert cluster.drop_redundant(terms) == ["language models", "rust"]


def test_concept_matcher_respects_word_boundaries():
    m = cluster.concept_matcher("ai")
    assert m.search("The AI act passes")
    assert not m.search("The air quality drops")
    bigram = cluster.concept_matcher("open source")
    assert bigram.search("New open  source release")
    assert not bigram.search("openly sourced")


def test_concept_shares_applies_floor():
    titles = ["ai story one", "ai story two", "database story", "another database"]
    labels = np.array([0, 0, 1, 1])
    shares = cluster.concept_shares(titles, labels, ["ai", "database", "rust"])
    assert shares[0] == {"ai": 1.0}
    assert shares[1] == {"database": 1.0}


def test_topic_terms_never_span_titles():
    """N-grams must come from single titles, not cluster concatenation."""
    titles = [
        "release of alpha",
        "beta version ships",
        "release of alpha again",
        "beta version delayed",
        "cats sleep all afternoon",
        "cats chase the laser",
        "cats ignore the laser",
        "release of alpha for cats",
    ]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    terms = cluster.topic_terms(titles, labels, top_n=6)
    flat = [t for ts in terms.values() for t in ts]
    assert "alpha beta" not in flat


def test_topic_similarity_symmetric_pairs():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 8))
    vectors = np.vstack([base[0], base[0] + 0.01, base[1], base[1] + 0.01, base[2]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 2])
    edges = cluster.topic_similarity(vectors, labels, top_n=1, floor=0.9)
    for a, b, w in edges:
        assert a < b
        assert 0.9 <= w <= 1.0
