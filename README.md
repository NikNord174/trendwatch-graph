# Trendwatch graph

[![ci](https://github.com/NikNord174/trendwatch-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/NikNord174/trendwatch-graph/actions/workflows/ci.yml)

An interactive trend-watching knowledge graph over six months of tech news:
11,969 Hacker News stories clustered into 98 topics, tethered to 30 shared
concepts, and scored week by week. I built the whole path myself, from
ingestion and change detection through embeddings, clustering, and layout to
the exploration UI. A frozen snapshot ships in the repo, so it runs anywhere
from a plain pip install: no database, no keys.

![Trend map](docs/trend_map.png)

| Timeline | Relations | Topics |
|---|---|---|
| ![Timeline](docs/timeline_map.png) | ![Relations](docs/relations_map.png) | ![Topics](docs/topics.png) |

## What you're looking at

- **Trend map** — the full graph: white dots are anchor concepts, coloured
  hubs are topics, stories attach to their topic. WebGL-rendered, click
  anything for its card.
- **Timeline** — the same map with a period filter; topics grow and shrink
  as you drag the date range.
- **Topics** — a searchable table with weekly-signal sparklines and a
  velocity score, plus a per-topic profile and story list.
- **Relations** — topics tethered to their anchor concepts, with links
  between similar topics.
- **Ontology** — the schema behind the graph and the snapshot's
  change-detection counters.
- **Chat** — questions answered over the snapshot: lexical retrieval picks
  topics and stories, an LLM writes the answer if a key is configured
  (without one the tab degrades to plain retrieval).

## Run it

Needs Python 3.11 or newer.

```
git clone https://github.com/NikNord174/trendwatch-graph
cd trendwatch-graph
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker: `make docker-build && make docker-run`, then open
http://localhost:8501. The map viewers load one rendering module from a CDN,
so viewing the maps needs internet.

## Rebuild the data

```
make install-pipeline
make data
```

`pipeline/fetch.py` pulls the last 180 days of stories with 50+ points from
the keyless Algolia HN API; `pipeline/run.py` deduplicates them by normalized
content hash, embeds titles locally (all-MiniLM-L6-v2), clusters them, labels
clusters c-TF-IDF style, lays the graph out with Fruchterman-Reingold, and
writes the snapshot. Each run diffs its hash set against the previous
snapshot, so the Ontology tab can report exactly which stories are new or
changed — the same mechanism an incremental ingest would use against a
database, applied to flat files.

Clustering defaults to Louvain community detection over a k-nearest-neighbour
similarity graph: 10 neighbours per story, edges below 0.30 cosine dropped,
and communities under 10 stories folded into the nearest surviving centroid.
The method picks its own topic count instead of taking one from me — the
committed snapshot's 98 topics come from `--resolution 3.5`, chosen after a
sweep from 25 topics at 1.0 to 184 at 8.0. Leiden is the known successor to
Louvain and would be the next step. `--method kmeans` keeps the old path with
`--k` for its topic count. `make data-cached` rebuilds from the cached raw
file without refetching — same corpus, so the Ontology tab reports a 0/0/0
diff against the previous snapshot — and `SAVE_VECTORS=1` also writes the
embeddings to `data/vectors.npz` (gitignored, rebuildable; the Postgres
loader below needs it).

## Optional backends

Flat files are the right store at this scale — the whole snapshot ships in
the repo and loads in well under a second. The loaders under `backends/`
exist to show the same data on query engines, Postgres with pgvector
and Neo4j, and what the load path looks like when the flat files are the
source of truth.

```
make install-backends
make backends-up     # pgvector on localhost:5433, Neo4j browser on :7474
make pg-load         # needs data/vectors.npz: make data-cached SAVE_VECTORS=1
make neo4j-load
make backends-down
```

Both databases run from `docker-compose.backends.yml` (credentials
`postgres`/`trendwatch` and `neo4j`/`trendwatch`, bolt on :7687); the demo
app never touches them. The Postgres loader finishes with a hybrid query:
more-like-this by embedding distance (`<=>`) with tier and date filters in
the same SQL. The Neo4j loader ships three Cypher queries: `concept_hubs`
(top concepts by anchored topics and total story reach, the default),
`bridges` (topic pairs a concept ties together that the similarity edges
missed), and `concept_path` (shortest path between two topic labels through
the concept layer, `--a`/`--b`).

## Design notes

- **Signal model.** A story contributes `tier_weight x (1 + log10(points/50))`
  to its topic's week. Points are the story's Hacker News score, the upvotes
  it collected; 50 is the fetch cutoff, so a story at the cutoff contributes
  exactly its tier weight, a 500-point story twice that, a 5,000-point story
  three times. The log damping is the point: one viral story cannot drown a
  quarter of steady coverage. `tier_weight` is the source-quality multiplier
  from the next bullet, 1.25 for primary sources down to 0.8 for social.
  Velocity is the last four weeks of signal minus the four before.
- **Source tiers.** Domains are ranked 1-4 (primary sources, established
  press, blogs, social/aggregators) by a small curated table. Crude, but
  honest and easy to audit.
- **Precomputed layout.** The force simulation runs offline; the viewers get
  final coordinates and keep the GPU for rendering only, which is why 12k
  nodes stay smooth.
- **One section per rerun.** Each map inlines a multi-megabyte data bundle,
  so the app renders one section at a time instead of embedding all maps at
  once.

## Tests

`make test` — 76 pytest cases: the pure pipeline functions (hashing, snapshot
diff, clustering helpers including the kNN graph and Louvain post-processing,
the signal model, chat retrieval ranking, eval scoring, the backend row
builders) and integrity checks over the committed snapshot itself (links
resolve, dates parse, hashes recompute, no orphaned concepts). Streamlit glue
is deliberately untested.

## Eval

`make eval` runs the golden questions in `eval/golden.json` through the same
retrieval path the Chat tab uses and prints hit@k — did a known-good story
make the top k — plus pairwise order agreement, the fraction of story pairs
the retrieval ranks in the same order as my reference list. It always exits
0: a measurement, not a CI gate. Three of the twelve questions (q02, q05,
q06) miss at every k; q02 is the best-understood miss — its topic ranks
first, but within-topic ranking buries the right story. All three stay in,
because a golden set that always hits measures nothing.

## Data and licensing

Story metadata (title, link, score, date) comes from the public
[Algolia Hacker News API](https://hn.algolia.com/api); all linked content
belongs to its original publishers. The snapshot is frozen at 2026-07-29.
Maps are rendered with [Cosmograph](https://cosmograph.app)
(`@cosmograph/cosmos`, CC-BY-NC-4.0, loaded from CDN). Code is MIT.

## Limitations

- Louvain has a resolution limit: genuinely small topics get absorbed into
  bigger ones, and the merge of sub-10-story communities amplifies that. The
  incoherent tail is smaller than k-means left, but it exists — visible in
  the Topics table, and I kept it; hiding a model's failure modes would
  misrepresent it.
- Velocity over a frozen snapshot is illustrative; in live operation it
  would be recomputed on every refresh.
- Concept extraction is document-frequency based, so anchors skew toward the
  vocabulary HN favours.
