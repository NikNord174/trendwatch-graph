"""Topic clustering, labelling, and anchor-concept extraction."""

import random
import re

import numpy as np

HN_PREFIX = re.compile(r"^(show|ask|tell|launch)\s+hn:?\s*", re.IGNORECASE)

# Terms that dominate HN titles without carrying topical meaning.
EXTRA_STOPWORDS = {
    "hn",
    "show",
    "ask",
    "new",
    "using",
    "use",
    "used",
    "make",
    "makes",
    "made",
    "like",
    "just",
    "way",
    "ways",
    "year",
    "years",
    "day",
    "days",
    "time",
    "times",
    "world",
    "based",
    "release",
    "released",
    "releases",
    "launch",
    "launches",
    "open",
    "source",
    "free",
    "vs",
    "best",
    "better",
    "faster",
    "inside",
    "million",
    "billion",
    "big",
    "small",
    "first",
    "says",
    "said",
    "getting",
    "built",
    "build",
    "building",
    "need",
    "needs",
    "really",
    "actually",
    "things",
    "thing",
    "people",
    # The token pattern requires a leading letter, so numerics and years never
    # tokenize; apostrophe forms do, and sklearn's list has none of them.
    "don't",
    "doesn't",
    "isn't",
    "won't",
    "can't",
    "didn't",
    "it's",
    "what's",
    "here's",
    "there's",
    "let's",
    "i'm",
    "you're",
    "we're",
}


def clean_title(title: str) -> str:
    return HN_PREFIX.sub("", title).strip()


def stopwords() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS)


def pick_k(n_stories: int) -> int:
    """Roughly one topic per ~90 stories, bounded to keep the map readable.

    Small corpora get k capped at n/3 so k-means always has room to fit;
    the 40..120 band only applies once the corpus can support it.
    """
    k = int(np.clip(n_stories // 90, 40, 120))
    return max(2, min(k, n_stories // 3 or 2))


def cluster_stories(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(vectors)


def knn_graph(
    vectors: np.ndarray, k_neighbors: int = 10, floor: float = 0.30
) -> list[tuple[int, int, float]]:
    """kNN edges over normalized vectors (cosine is a dot product here),
    computed in row blocks: the full similarity matrix for 12k stories would
    be ~570 MB float32, while 1000-row blocks keep the peak around 50 MB."""
    n = len(vectors)
    k = min(k_neighbors, n - 1)  # tiny corpora: cannot rank more neighbors than exist
    if k < 1:
        return []  # with 0 or 1 stories the [-k:] slice would take every column, self included
    weights: dict[tuple[int, int], float] = {}
    for start in range(0, n, 1000):
        block = vectors[start : start + 1000] @ vectors.T
        for offset, sims in enumerate(block):
            i = start + offset
            sims[i] = -np.inf  # exclude self; -1.0 could tie with an antipodal neighbor
            for j in np.argpartition(sims, -k)[-k:]:
                w = float(sims[j])
                if w >= floor:
                    pair = (min(i, int(j)), max(i, int(j)))  # union-symmetrize
                    weights[pair] = max(weights.get(pair, w), w)  # 0.0 would beat a negative w
    # Sorted edges make reruns byte-identical: Louvain membership is
    # sensitive to edge order.
    return [(i, j, weights[(i, j)]) for i, j in sorted(weights)]


def louvain_labels(
    edges: list[tuple[int, int, float]], n_nodes: int, seed: int = 42, resolution: float = 1.0
) -> np.ndarray:
    import igraph as ig  # heavy dep, only the louvain path needs it

    # igraph draws from Python's random module, not numpy's — same idiom as
    # layout.layout_positions. Isolated nodes come out as singleton
    # communities; merge_small absorbs them.
    ig.set_random_number_generator(random.Random(seed))
    graph = ig.Graph(n=n_nodes, edges=[(a, b) for a, b, _ in edges])
    part = graph.community_multilevel(weights=[w for _, _, w in edges], resolution=resolution)
    return np.asarray(part.membership, dtype=int)


def merge_small(labels: np.ndarray, vectors: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Tiny communities render as unreadable map blobs; fold each member into
    the cosine-nearest surviving community by centroid."""
    ids, sizes = np.unique(labels, return_counts=True)
    survivors = ids[sizes >= min_size]
    if len(survivors) == 0 or len(survivors) == len(ids):
        return labels  # all-small (degenerate) or nothing to merge
    centroids = np.stack([vectors[labels == t].mean(axis=0) for t in survivors])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    out = labels.copy()
    small = ~np.isin(labels, survivors)
    out[small] = survivors[np.argmax(vectors[small] @ centroids.T, axis=1)]
    return out


def relabel_dense(labels: np.ndarray) -> np.ndarray:
    """Dense 0..k-1 ids ordered by descending size, so topic 0 is always the
    biggest and exports stay readable."""
    ids, sizes = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-sizes, kind="stable")]  # size ties break by lower old id
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.asarray([mapping[int(t)] for t in labels], dtype=int)


def _fold_plural(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def overlaps(a: str, b: str) -> bool:
    """Whole-word containment with plural folding: 'models' overlaps 'model'
    and 'language models', but 'ai' does not overlap 'openai' and 'war' does
    not overlap 'software'."""
    wa = [_fold_plural(w) for w in a.split()]
    wb = [_fold_plural(w) for w in b.split()]
    short, long = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    return any(long[i : i + len(short)] == short for i in range(len(long) - len(short) + 1))


def drop_redundant(terms: list[str]) -> list[str]:
    """Remove terms already contained in an earlier, higher-ranked term."""
    kept: list[str] = []
    for term in terms:
        if any(overlaps(term, other) for other in kept):
            continue
        kept.append(term)
    return kept


def topic_terms(titles: list[str], labels: np.ndarray, top_n: int = 6) -> dict[int, list[str]]:
    """Characteristic terms per cluster, c-TF-IDF style.

    Titles are vectorized individually and the counts summed per cluster, so
    n-grams never span two titles; the score then rewards terms frequent in a
    cluster but present in few clusters.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        stop_words=stopwords(),
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
        token_pattern=r"[a-zA-Z][a-zA-Z0-9+#.'-]+",
    )
    matrix = vec.fit_transform(clean_title(t).lower() for t in titles)
    vocab = vec.get_feature_names_out()
    ids = sorted({int(t) for t in labels})

    cluster_tf = np.stack(
        [np.asarray(matrix[labels == t].sum(axis=0)).ravel() for t in ids]
    ).astype(np.float64)
    df_clusters = (cluster_tf > 0).sum(axis=0)
    weight = np.log(1 + len(ids) / (1 + df_clusters))
    totals = cluster_tf.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    scores = cluster_tf / totals * weight

    out: dict[int, list[str]] = {}
    for row, t in enumerate(ids):
        ranked = [vocab[i] for i in scores[row].argsort()[::-1] if scores[row][i] > 0]
        out[t] = drop_redundant(ranked[: top_n * 3])[:top_n]
    return out


def topic_label(terms: list[str]) -> str:
    return " · ".join(terms[:3]) if terms else "unlabelled"


def extract_concepts(titles: list[str], max_concepts: int = 30) -> list[tuple[str, int]]:
    """Anchor concepts: the most widespread terms across the whole corpus.

    Document frequency, not TF-IDF — anchors should be common ground between
    topics, which is the opposite of what TF-IDF rewards.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        stop_words=stopwords(),
        ngram_range=(1, 2),
        binary=True,
        min_df=20,
        token_pattern=r"[a-zA-Z][a-zA-Z0-9+#.'-]+",
    )
    matrix = vec.fit_transform(clean_title(t).lower() for t in titles)
    df = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    ranked = sorted(zip(vocab, df), key=lambda x: -x[1])
    picked: list[tuple[str, int]] = []
    for term, count in ranked:
        if any(overlaps(term, seen) for seen, _ in picked):
            continue
        picked.append((term, int(count)))
        if len(picked) >= max_concepts:
            break
    return picked


def concept_matcher(concept: str) -> re.Pattern:
    """Lookarounds instead of \\b: the tokenizer produces terms ending in
    non-word chars ('u.s.', 'c++'), and a trailing \\b after those can never
    match, which would silently orphan the concept in the exported graph."""
    words = [re.escape(w) for w in concept.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


def concept_shares(
    titles: list[str], labels: np.ndarray, concepts: list[str]
) -> dict[int, dict[str, float]]:
    """Per topic: fraction of its stories that mention each anchor concept."""
    matchers = {c: concept_matcher(c) for c in concepts}
    by_topic: dict[int, list[str]] = {}
    for title, lab in zip(titles, labels):
        by_topic.setdefault(int(lab), []).append(title)
    shares: dict[int, dict[str, float]] = {}
    for t, members in by_topic.items():
        counts = {c: sum(1 for title in members if m.search(title)) for c, m in matchers.items()}
        shares[t] = {
            c: round(n / len(members), 4) for c, n in counts.items() if n / len(members) >= 0.08
        }
    return shares


def topic_similarity(
    vectors: np.ndarray, labels: np.ndarray, top_n: int = 3, floor: float = 0.5
) -> list[tuple[int, int, float]]:
    """Strongest centroid-cosine pairs between topics, for the relations map."""
    ids = sorted({int(t) for t in labels})
    centroids = np.stack([vectors[labels == t].mean(axis=0) for t in ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = centroids @ centroids.T
    edges: set[tuple[int, int]] = set()
    out: list[tuple[int, int, float]] = []
    for i, t in enumerate(ids):
        order = np.argsort(sim[i])[::-1]
        added = 0
        for j in order:
            if j == i or added >= top_n:
                continue
            if sim[i, j] < floor:
                break
            pair = (min(t, ids[j]), max(t, ids[j]))
            if pair not in edges:
                edges.add(pair)
                out.append((pair[0], pair[1], float(sim[i, j])))
            added += 1
    return out


def dominant_area(shares: dict[str, float]) -> str:
    return max(shares, key=shares.get) if shares else "general"
