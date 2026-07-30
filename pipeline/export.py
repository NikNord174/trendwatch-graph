"""Assemble the graph and write the snapshot the app runs on.

Outputs:
  assets/graph_data.js      window.GRAPH_DATA for the trend and timeline maps
  assets/relations_data.js  window.TREND_DATA for the relations map
  data/topics.json          topic profiles for the Topics tab
  data/news_topics.csv      per-story rows for the signal model
  data/hashes.json          content hashes + diff vs the previous snapshot
"""

import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from . import cluster, layout

TIER_PRIMARY = {
    "github.com",
    "gitlab.com",
    "arxiv.org",
    "kernel.org",
    "lwn.net",
    "python.org",
    "postgresql.org",
    "sqlite.org",
    "mozilla.org",
    "golang.org",
    "rust-lang.org",
    "llvm.org",
    "debian.org",
    "w3.org",
    "ietf.org",
}
TIER_PRESS = {
    "arstechnica.com",
    "theverge.com",
    "wired.com",
    "techcrunch.com",
    "reuters.com",
    "bloomberg.com",
    "nytimes.com",
    "theguardian.com",
    "bbc.com",
    "bbc.co.uk",
    "ft.com",
    "economist.com",
    "washingtonpost.com",
    "wsj.com",
    "theregister.com",
    "zdnet.com",
    "spectrum.ieee.org",
    "nature.com",
    "science.org",
    "apnews.com",
    "cnbc.com",
    "404media.co",
}
TIER_WEAK = {
    "medium.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "dev.to",
    "threads.net",
    "tiktok.com",
}

AREA_PALETTE = [
    "#8a93ff",
    "#58c9a5",
    "#f2a65a",
    "#e46a8a",
    "#59b5e8",
    "#c98ff2",
    "#a5c95a",
    "#f2d45a",
    "#f27e5a",
    "#6ae4c9",
    "#e46ad4",
    "#9aa7b8",
]


def domain_of(url: str) -> str:
    if not url:
        return "news.ycombinator.com"
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def source_tier(domain: str) -> int:
    if domain in TIER_PRIMARY or domain.endswith((".gov", ".edu")):
        return 1
    if domain in TIER_PRESS:
        return 2
    if domain in TIER_WEAK or domain.endswith(".substack.com"):
        return 4
    return 3


def importance(points: int) -> str:
    if points >= 300:
        return "High"
    if points >= 120:
        return "Medium"
    return "Low"


def slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def md_link(title: str, url: str) -> str:
    """Link that stays parseable by the viewer's simple linkifier: square
    brackets in titles become parens, parens in URLs get percent-encoded."""
    text = title.replace("[", "(").replace("]", ")")
    return f"[{text}]({url.replace('(', '%28').replace(')', '%29')})"


def build_graph(
    stories: list[dict],
    labels: np.ndarray,
    terms: dict[int, list[str]],
    concepts: list[tuple[str, int]],
    shares: dict[int, dict[str, float]],
) -> tuple[list[dict], list[dict]]:
    """Nodes and links for the trend/timeline maps, with positions attached."""
    matchers = {c: cluster.concept_matcher(c) for c, _ in concepts}
    topic_sizes = Counter(int(t) for t in labels)

    # One matcher pass per story feeds kw tags, co-occurrence edges, and the
    # concept node sizes, so the three can never disagree.
    matched = [sorted(c for c, m in matchers.items() if m.search(s["title"])) for s in stories]
    concept_size = Counter(c for found in matched for c in found)

    nodes: list[dict] = []
    for term, _ in concepts:
        nodes.append(
            {"id": f"c:{slug(term)}", "t": "concept", "label": term, "size": concept_size[term]}
        )
    for t in sorted(topic_sizes):
        nodes.append(
            {
                "id": f"t:{t}",
                "t": "topic",
                "topic": t,
                "label": cluster.topic_label(terms.get(t, [])),
                "size": topic_sizes[t],
                "area": cluster.dominant_area(shares.get(t, {})),
            }
        )
    for story, lab, found in zip(stories, labels, matched):
        domain = domain_of(story["url"])
        nodes.append(
            {
                "id": f"s:{story['id']}",
                "t": "story",
                "label": story["title"],
                "url": story["url"],
                "domain": domain,
                "date": story["created_at"][:10],
                "points": story["points"],
                "comments": story["comments"],
                "tier": source_tier(domain),
                "imp": importance(story["points"]),
                "topic": int(lab),
                "kw": found[:3],
            }
        )

    index = {n["id"]: i for i, n in enumerate(nodes)}
    links: list[dict] = []
    edges: list[tuple[int, int, float]] = []

    def add(a: str, b: str, rel: str, w: float, weight: float) -> None:
        link = {"source": a, "target": b, "rel": rel}
        if rel == "topic_concept":
            link["w"] = round(w, 3)
        links.append(link)
        edges.append((index[a], index[b], weight))

    for t, share_map in shares.items():
        for c, share in sorted(share_map.items(), key=lambda x: -x[1])[:4]:
            add(f"t:{t}", f"c:{slug(c)}", "topic_concept", share, 2.0 + 3.0 * share)

    cooc = Counter()
    for found in matched:
        cooc.update(combinations(found, 2))
    for (a, b), n in cooc.most_common(40):
        add(f"c:{slug(a)}", f"c:{slug(b)}", "related", 0.0, 0.5 + min(n / 50, 1.0))

    for story, lab in zip(stories, labels):
        add(f"s:{story['id']}", f"t:{int(lab)}", "in_topic", 0.0, 1.0)

    positions = layout.layout_positions(len(nodes), edges)
    for node, (x, y) in zip(nodes, positions):
        node["x"] = round(float(x), 1)
        node["y"] = round(float(y), 1)
    return nodes, links


def topic_report(
    t: int,
    label: str,
    terms: list[str],
    members: list[dict],
) -> str:
    dates = sorted(m["date"] for m in members if m["date"])
    span = f"{dates[0]} to {dates[-1]}" if dates else "n/a"
    domains = Counter(m["domain"] for m in members).most_common(4)
    top = sorted(members, key=lambda m: -m["points"])[:5]
    lines = [
        f"Auto-generated profile of topic {t} ({label}).",
        "",
        f"- {len(members)} stories, {span}",
        f"- defining terms: {', '.join(terms[:6])}",
        f"- main sources: {', '.join(d for d, _ in domains)}",
        "",
        "Top stories by score:",
    ]
    for i, m in enumerate(top, 1):
        lines.append(f"{i}. {md_link(m['label'], m['url'])}, {m['points']} points, {m['date']}")
    return "\n".join(lines)


def build_relations(
    graph_nodes: list[dict],
    terms: dict[int, list[str]],
    shares: dict[int, dict[str, float]],
    topic_sim: list[tuple[int, int, float]],
    hub_count: int = 10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Nodes, links, and legend areas for the relations map."""
    concepts = [n for n in graph_nodes if n["t"] == "concept"]
    hubs = sorted(concepts, key=lambda n: -n["size"])[:hub_count]
    hub_color = {n["label"]: AREA_PALETTE[i % len(AREA_PALETTE)] for i, n in enumerate(hubs)}
    areas = [{"name": name, "color": color} for name, color in hub_color.items()]

    stories_by_topic: dict[int, list[dict]] = {}
    for n in graph_nodes:
        if n["t"] == "story":
            stories_by_topic.setdefault(n["topic"], []).append(n)

    nodes: list[dict] = []
    for hub in hubs:
        nodes.append(
            {
                "id": hub["id"],
                "t": "concept",
                "label": hub["label"],
                "size": hub["size"],
                "color": hub_color[hub["label"]],
            }
        )
    for n in graph_nodes:
        if n["t"] != "topic":
            continue
        t = n["topic"]
        members = stories_by_topic.get(t, [])
        reliable = [m for m in members if m["tier"] <= 2] or members
        top_news = sorted(reliable, key=lambda m: -m["points"])[:8]
        area = n["area"] if n["area"] in hub_color else None
        nodes.append(
            {
                "id": n["id"],
                "t": "topic",
                "topic": t,
                "label": n["label"],
                "size": n["size"],
                "area": area or "",
                "color": hub_color.get(area, "#9aa7b8"),
                "news": [
                    {
                        "title": m["label"],
                        "url": m["url"],
                        "date": m["date"],
                        "points": m["points"],
                        "tier": m["tier"],
                    }
                    for m in top_news
                ],
                "news_total": len(members),
                "report": topic_report(t, n["label"], terms.get(t, []), members),
            }
        )

    index = {n["id"]: i for i, n in enumerate(nodes)}
    links: list[dict] = []
    edges: list[tuple[int, int, float]] = []
    for t, share_map in shares.items():
        for c, share in sorted(share_map.items(), key=lambda x: -x[1]):
            cid = f"c:{slug(c)}"
            if cid in index and f"t:{t}" in index:
                links.append(
                    {
                        "source": f"t:{t}",
                        "target": cid,
                        "rel": "t_c",
                        "color": hub_color.get(c, "#9aa7b8"),
                    }
                )
                edges.append((index[f"t:{t}"], index[cid], 1.0 + 3.0 * share))
                break  # one hub tether per topic keeps the map legible
    for a, b, w in topic_sim:
        if f"t:{a}" in index and f"t:{b}" in index:
            links.append({"source": f"t:{a}", "target": f"t:{b}", "rel": "t_t"})
            edges.append((index[f"t:{a}"], index[f"t:{b}"], w))

    positions = layout.layout_positions(len(nodes), edges)
    for node, (x, y) in zip(nodes, positions):
        node["x"] = round(float(x), 1)
        node["y"] = round(float(y), 1)
    return nodes, links, areas


def write_js(path: Path, var: str, payload: dict) -> None:
    path.write_text(
        f"window.{var} = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def write_topics_json(
    path: Path,
    graph_nodes: list[dict],
    relation_nodes: list[dict],
    terms: dict[int, list[str]],
    shares: dict[int, dict[str, float]],
) -> None:
    reports = {n["topic"]: n["report"] for n in relation_nodes if n["t"] == "topic"}
    rows = []
    for n in graph_nodes:
        if n["t"] != "topic":
            continue
        t = n["topic"]
        rows.append(
            {
                "id": t,
                "label": n["label"],
                "area": n["area"],
                "size": n["size"],
                "terms": terms.get(t, []),
                "anchors": shares.get(t, {}),
                "report": reports.get(t, ""),
            }
        )
    rows.sort(key=lambda r: -r["size"])
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def write_news_csv(path: Path, graph_nodes: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "topic", "date", "points", "tier", "domain", "title", "url"])
        for n in graph_nodes:
            if n["t"] == "story":
                writer.writerow(
                    [
                        n["id"],
                        n["topic"],
                        n["date"],
                        n["points"],
                        n["tier"],
                        n["domain"],
                        n["label"],
                        n["url"],
                    ]
                )
