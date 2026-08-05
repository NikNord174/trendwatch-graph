"""The graph ships as flat files so the demo needs no services; loading the
same snapshot into Neo4j makes the concept layer queryable multi-hop, and the
bundled queries answer questions the static map views cannot."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

CONSTRAINTS = (
    "CREATE CONSTRAINT story_id IF NOT EXISTS FOR (s:Story) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
)  # unindexed MERGE label-scans on every row, turning 12k stories into minutes

CYPHER = {  # insertion order is load order: nodes first, so the edge MATCHes can hit
    "Story": (
        "UNWIND $rows AS row MERGE (s:Story {id: row.id}) "
        "SET s.title = row.title, s.url = row.url, s.date = date(row.date), "
        "s.points = row.points, s.tier = row.tier, s.domain = row.domain"
    ),
    "Topic": (
        "UNWIND $rows AS row MERGE (t:Topic {id: row.id}) "
        "SET t.label = row.label, t.area = row.area, t.size = row.size"
    ),
    "Concept": "UNWIND $rows AS row MERGE (:Concept {name: row.name})",
    "IN_TOPIC": (
        "UNWIND $rows AS row MATCH (s:Story {id: row.story}), (t:Topic {id: row.topic}) "
        "MERGE (s)-[:IN_TOPIC]->(t)"
    ),
    "ANCHORED_BY": (
        "UNWIND $rows AS row MATCH (t:Topic {id: row.topic}), (c:Concept {name: row.concept}) "
        "MERGE (t)-[r:ANCHORED_BY]->(c) SET r.weight = row.weight"
    ),
    "SIMILAR_TO": (
        "UNWIND $rows AS row MATCH (a:Topic {id: row.a}), (b:Topic {id: row.b}) "
        "MERGE (a)-[r:SIMILAR_TO]-(b) "  # undirected: one edge per pair either way round
        "SET r.weight = row.weight"
    ),
    "RELATED": (
        "UNWIND $rows AS row MATCH (a:Concept {name: row.a}), (b:Concept {name: row.b}) "
        "MERGE (a)-[:RELATED]-(b)"
    ),
}

DOCUMENTED_QUERIES = {
    "concept_hubs": (  # concepts anchoring the most topics, with their total story reach
        "MATCH (c:Concept)<-[:ANCHORED_BY]-(t:Topic)<-[:IN_TOPIC]-(s:Story) "
        "RETURN c.name AS concept, count(DISTINCT t) AS topics, "
        "count(DISTINCT s) AS stories ORDER BY topics DESC, stories DESC LIMIT 10"
    ),
    "bridges": (  # topic pairs a shared anchor ties together that the similarity edges missed
        "MATCH (a:Topic)-[:ANCHORED_BY]->(c:Concept)<-[:ANCHORED_BY]-(b:Topic) "
        "WHERE a.id < b.id AND NOT (a)-[:SIMILAR_TO]-(b) "
        "RETURN a.label AS topic_a, b.label AS topic_b, collect(c.name) AS via "
        "ORDER BY size(via) DESC LIMIT 15"
    ),
    "concept_path": (  # how two topic labels connect through the concept layer ($a, $b)
        "MATCH (a:Topic {label: $a}), (b:Topic {label: $b}), "
        "p = shortestPath((a)-[:ANCHORED_BY|RELATED|SIMILAR_TO*..8]-(b)) "
        "RETURN [n IN nodes(p) | coalesce(n.label, n.name)] AS path"
    ),
}


def _read_js(path: Path) -> dict:
    """Map bundles are `window.X = {...};` scripts, so the payload starts after
    the first `=` — a standalone copy of the app's trick, because the backends
    must not drag in Streamlit-cached modules."""
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw.partition("=")[2].strip().rstrip(";"))


def node_rows(news_df: pd.DataFrame, topics: list[dict], graph: dict) -> dict[str, list[dict]]:
    """UNWIND-ready rows per node label; int() casts strip the numpy scalars
    the driver refuses to serialise, and layout-only bundle fields stay out."""
    return {
        "Story": [
            {
                "id": rec.id,
                "title": rec.title,
                "url": None if pd.isna(rec.url) else rec.url,  # a null SET skips the property
                "date": str(rec.date)[:10],  # csv gives ISO strings, test frames give Timestamps
                "points": int(rec.points),
                "tier": int(rec.tier),
                "domain": rec.domain,
            }
            for rec in news_df.itertuples(index=False)
        ],
        "Topic": [
            {"id": int(t["id"]), "label": t["label"], "area": t["area"], "size": int(t["size"])}
            for t in topics
        ],
        "Concept": [{"name": n["label"]} for n in graph["nodes"] if n["t"] == "concept"],
    }


def edge_rows(news_df: pd.DataFrame, graph: dict, relations: dict) -> dict[str, list[dict]]:
    """UNWIND-ready rows per relationship type; concept link ends go through the
    bundle's id->label map because slugged ids ('c:u-s' vs 'u.s.') do not
    survive plain prefix stripping."""
    concept = {n["id"]: n["label"] for n in graph["nodes"] if n["t"] == "concept"}
    return {
        "IN_TOPIC": [
            {"story": rec.id, "topic": int(rec.topic)} for rec in news_df.itertuples(index=False)
        ],
        "ANCHORED_BY": [
            {
                "topic": int(link["source"].removeprefix("t:")),
                "concept": concept[link["target"]],
                "weight": link.get("w"),  # absent weight stays null, and SET null is a no-op
            }
            for link in graph["links"]
            if link["rel"] == "topic_concept"
        ],
        "SIMILAR_TO": [
            {
                "a": int(link["source"].removeprefix("t:")),
                "b": int(link["target"].removeprefix("t:")),
                "weight": link.get("w"),
            }
            for link in relations["links"]
            if link["rel"] == "t_t"  # t_c links tie topics to concepts, not to each other
        ],
        "RELATED": [
            {"a": concept[link["source"]], "b": concept[link["target"]]}
            for link in graph["links"]
            if link["rel"] == "related"
        ],
    }


def _run_batched(session, cypher: str, rows: list[dict], size: int = 1000) -> None:
    """The whole UNWIND list sits in transaction memory at once, so 12k story
    rows go over in slices rather than one giant parameter."""
    for start in range(0, len(rows), size):
        session.run(cypher, rows=rows[start : start + size]).consume()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--auth", default="neo4j/trendwatch", help="user/password, as in compose")
    parser.add_argument("--query", choices=sorted(DOCUMENTED_QUERIES), default="concept_hubs")
    parser.add_argument("--a", help="start topic label for concept_path")
    parser.add_argument("--b", help="end topic label for concept_path")
    args = parser.parse_args()
    if args.query == "concept_path" and not (args.a and args.b):
        parser.error("concept_path needs --a and --b topic labels")

    import neo4j  # deferred so the app and base CI never need the driver installed

    news = pd.read_csv(DATA / "news_topics.csv")
    topics = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))
    graph = _read_js(ASSETS / "graph_data.js")
    relations = _read_js(ASSETS / "relations_data.js")
    rows = node_rows(news, topics, graph) | edge_rows(news, graph, relations)

    user, _, password = args.auth.partition("/")
    driver = neo4j.GraphDatabase.driver(args.uri, auth=(user, password))
    with driver, driver.session() as session:
        for statement in CONSTRAINTS:
            session.run(statement).consume()
        for name, cypher in CYPHER.items():
            _run_batched(session, cypher, rows[name])
        print(f"merged {sum(len(v) for v in rows.values())} rows; running {args.query}:")
        result = session.run(DOCUMENTED_QUERIES[args.query], a=args.a, b=args.b)
        for record in result:
            print(" | ".join(str(value) for value in record.values()))


if __name__ == "__main__":
    main()
