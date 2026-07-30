"""Run the processing pipeline: raw stories -> committed snapshot.

Usage:
    python -m pipeline.fetch    # refresh pipeline/raw/stories.jsonl
    python -m pipeline.run      # everything else, writes assets/ and data/
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from . import cluster, dedupe, embed, export

ROOT = Path(__file__).resolve().parent.parent


def load_raw(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(__file__).parent / "raw" / "stories.jsonl")
    parser.add_argument("--k", type=int, default=0, help="topic count; 0 = auto")
    args = parser.parse_args()

    stories = dedupe.dedupe_stories(load_raw(args.raw))
    print(f"{len(stories)} stories after dedupe")

    hashes = dedupe.hash_stories(stories)
    hashes_path = ROOT / "data" / "hashes.json"
    previous = dedupe.load_previous_hashes(hashes_path)
    diff = dedupe.diff_hashes(hashes, previous)
    print(
        f"vs previous snapshot: {len(diff['new'])} new, "
        f"{len(diff['changed'])} changed, {len(diff['removed'])} removed"
    )

    titles = [s["title"] for s in stories]
    vectors = embed.embed_titles(titles)

    k = args.k or cluster.pick_k(len(stories))
    labels = cluster.cluster_stories(vectors, k)
    terms = cluster.topic_terms(titles, labels)
    concepts = cluster.extract_concepts(titles)
    shares = cluster.concept_shares(titles, labels, [c for c, _ in concepts])
    sim = cluster.topic_similarity(vectors, labels)
    print(f"{k} topics, {len(concepts)} concepts, {len(sim)} topic-topic edges")

    nodes, links = export.build_graph(stories, labels, terms, concepts, shares)
    rel_nodes, rel_links, areas = export.build_relations(nodes, terms, shares, sim)

    export.write_js(
        ROOT / "assets" / "graph_data.js", "GRAPH_DATA", {"nodes": nodes, "links": links}
    )
    export.write_js(
        ROOT / "assets" / "relations_data.js",
        "TREND_DATA",
        {"nodes": rel_nodes, "links": rel_links, "areas": areas},
    )
    export.write_topics_json(ROOT / "data" / "topics.json", nodes, rel_nodes, terms, shares)
    export.write_news_csv(ROOT / "data" / "news_topics.csv", nodes)

    dates = sorted(s["created_at"][:10] for s in stories)
    hashes_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "window": {"from": dates[0], "to": dates[-1]},
                "stories": len(stories),
                "topics": k,
                "changes_vs_previous": {key: len(val) for key, val in diff.items()},
                "changed_ids": {"new": diff["new"][:200], "changed": diff["changed"][:200]},
                "hashes": hashes,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        "snapshot written: assets/graph_data.js, assets/relations_data.js, "
        "data/topics.json, data/news_topics.csv, data/hashes.json"
    )


if __name__ == "__main__":
    main()
