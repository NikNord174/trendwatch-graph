import numpy as np
import pytest

from pipeline import cluster


def _unit(angles_deg: list[float]) -> np.ndarray:
    """2D unit vectors at given angles, so cosine between any two is exactly
    the cosine of their angle difference and expectations stay readable."""
    rad = np.deg2rad(angles_deg)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


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


def test_knn_graph_pairs_ordered_and_sorted():
    edges = cluster.knn_graph(_unit([0, 10, 20, 90, 100, 180]), k_neighbors=2)
    pairs = [(i, j) for i, j, _ in edges]
    assert all(i < j for i, j in pairs)
    assert pairs == sorted(pairs)  # stable edge order in, stable Louvain out
    assert len(pairs) == len(set(pairs))  # union-symmetrization never duplicates


def test_knn_graph_floor_drops_weak_edges():
    edges = cluster.knn_graph(_unit([0, 5, 90]), k_neighbors=2, floor=0.30)
    assert {(i, j) for i, j, _ in edges} == {(0, 1)}  # cos(5 deg) ~ 0.996 survives
    assert edges[0][2] > 0.99


def test_knn_graph_union_symmetrizes():
    """With k=1, node 0 picks node 1 but node 1 picks node 2; the (0, 1)
    edge must survive — intersection-symmetrizing would orphan node 0."""
    edges = cluster.knn_graph(_unit([0, 10, 14]), k_neighbors=1, floor=0.30)
    assert [(i, j) for i, j, _ in edges] == [(0, 1), (1, 2)]


def test_knn_graph_edge_count_bounded():
    """Union-symmetrization can push a hub's degree past k_neighbors, but the
    total can never exceed n * k_neighbors: every undirected edge originates
    from at least one directed top-k selection."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    edges = cluster.knn_graph(vectors, k_neighbors=3, floor=-1.0)
    assert 0 < len(edges) <= 40 * 3


def test_knn_graph_deterministic():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(30, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert cluster.knn_graph(vectors) == cluster.knn_graph(vectors)


def test_knn_graph_single_story():
    assert cluster.knn_graph(_unit([0.0])) == []
    assert cluster.knn_graph(np.zeros((0, 2), dtype=np.float32)) == []


def test_knn_graph_keeps_negative_weights():
    edges = cluster.knn_graph(_unit([0, 180]), k_neighbors=1, floor=-1.0)
    assert edges == [(0, 1, -1.0)]  # cos(180 deg); a max seeded at 0.0 would report 0.0


def test_merge_small_absorbs_tiny_community():
    vectors = _unit([0, 5, 10, 12, 90, 95, 100, 102, 7, 8])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    merged = cluster.merge_small(labels, vectors, min_size=3)
    assert list(merged[:8]) == list(labels[:8])
    assert list(merged[8:]) == [0, 0]  # 7-8 degrees sit inside the 0-12 group


def test_merge_small_all_small_unchanged():
    vectors = _unit([0, 10, 90, 100])
    labels = np.array([0, 0, 1, 1])
    assert np.array_equal(cluster.merge_small(labels, vectors, min_size=3), labels)


def test_relabel_dense_orders_by_size():
    out = cluster.relabel_dense(np.array([7, 7, 7, 5, 2, 2]))
    assert list(out) == [0, 0, 0, 2, 1, 1]
    assert sorted(set(out.tolist())) == [0, 1, 2]  # dense 0..k-1, no gaps


def test_relabel_dense_breaks_ties_by_old_id():
    assert list(cluster.relabel_dense(np.array([5, 5, 2, 2, 8]))) == [1, 1, 0, 0, 2]


def test_louvain_labels_splits_two_cliques():
    pytest.importorskip("igraph")
    edges = [
        (i, j, 1.0) for group in (range(5), range(5, 10)) for i in group for j in group if i < j
    ]
    edges.append((4, 5, 0.1))  # weak bridge between the cliques
    edges.sort()
    first = cluster.louvain_labels(edges, 10)
    assert np.array_equal(first, cluster.louvain_labels(edges, 10))
    assert len(set(first.tolist())) == 2
    assert len(set(first[:5].tolist())) == 1
    assert len(set(first[5:].tolist())) == 1
    assert first[0] != first[5]


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
