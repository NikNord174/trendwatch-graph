import numpy as np

from pipeline import cluster


def test_clean_title_strips_hn_prefixes():
    assert cluster.clean_title("Show HN: My side project") == "My side project"
    assert cluster.clean_title("ask hn Why is this") == "Why is this"
    assert cluster.clean_title("Plain title") == "Plain title"


def test_pick_k_bounds():
    assert cluster.pick_k(9_000) == 100
    assert cluster.pick_k(100_000) == 120
    assert cluster.pick_k(100) == 33  # small corpora: k capped at n/3
    assert cluster.pick_k(10) == 3
    assert cluster.pick_k(2) == 2


def test_drop_redundant_removes_contained_terms():
    terms = ["language models", "models", "rust", "rust compiler"]
    assert cluster.drop_redundant(terms) == ["language models", "rust"]


def test_drop_redundant_keeps_subword_lookalikes():
    """Containment is whole-word: 'ai' must not swallow 'openai'."""
    assert cluster.drop_redundant(["ai", "openai"]) == ["ai", "openai"]
    assert cluster.drop_redundant(["software", "war"]) == ["software", "war"]
    assert cluster.drop_redundant(["apple", "app"]) == ["apple", "app"]


def test_concept_matcher_respects_word_boundaries():
    m = cluster.concept_matcher("ai")
    assert m.search("The AI act passes")
    assert not m.search("The air quality drops")
    assert not m.search("openai ships a model")
    bigram = cluster.concept_matcher("open source")
    assert bigram.search("New open  source release")
    assert not bigram.search("openly sourced")


def test_concept_matcher_handles_nonword_edges():
    """Terms the tokenizer produces can end in '.', '+', '#'; a plain \\b
    would never match them and the concept would ship orphaned."""
    assert cluster.concept_matcher("u.s.").search("New U.S. export rules")
    assert cluster.concept_matcher("c++").search("Writing C++ in 2026")
    assert not cluster.concept_matcher("c++").search("The c+ grade")


def test_concept_shares_applies_floor():
    titles = ["ai story one", "ai story two", "database story", "another database"]
    labels = np.array([0, 0, 1, 1])
    shares = cluster.concept_shares(titles, labels, ["ai", "database", "rust"])
    assert shares[0] == {"ai": 1.0}
    assert shares[1] == {"database": 1.0}


def test_concept_shares_floor_drops_nonzero_below_threshold():
    """One mention in twenty stories (share 0.05) sits below the 0.08 floor."""
    titles = ["ai one"] + [f"plain {i}" for i in range(19)]
    labels = np.array([0] * 20)
    shares = cluster.concept_shares(titles, labels, ["ai", "plain"])
    assert "ai" not in shares[0]
    assert shares[0]["plain"] == 0.95


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


def test_topic_similarity_finds_close_pair_once():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2, 8))
    # Clusters 0 and 1 are near-copies of each other; cluster 2 is independent.
    vectors = np.vstack([base[0], base[0] + 0.01, base[0] + 0.02, base[0] + 0.03, base[1]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 2])
    edges = cluster.topic_similarity(vectors, labels, top_n=1, floor=0.9)
    assert [(a, b) for a, b, _ in edges] == [(0, 1)]  # once, ordered, not mirrored
    assert 0.9 <= edges[0][2] <= 1.0
