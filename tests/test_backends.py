"""Backend loaders keep their builders pure; connections never happen in tests."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from backends import neo4j_loader, pg_loader

NEWS = pd.DataFrame(
    [
        {
            "id": "s:1",
            "topic": 7,
            "date": "2026-02-01",
            "points": 42,
            "tier": 2,
            "domain": "example.com",
            "title": "hello",
            "url": "https://example.com/x",
        },
        {
            "id": "s:2",
            "topic": 0,
            "date": "2026-02-02",
            "points": 61,
            "tier": 3,
            "domain": "news.ycombinator.com",
            "title": "Ask HN: where did my link go?",
            "url": np.nan,  # csv url-less rows arrive as float NaN, not as ""
        },
    ]
)

TOPICS = [
    {"id": 0, "label": "alpha · beta", "area": "ai", "size": 3},
    {"id": 7, "label": "gamma · delta", "area": "infra", "size": 2},
]

GRAPH = {
    "nodes": [
        {"id": "s:1", "t": "story", "label": "hello", "x": 1.0, "y": 2.0},
        {"id": "t:0", "t": "topic", "label": "alpha · beta", "size": 3},
        {"id": "c:ai", "t": "concept", "label": "ai", "size": 9},
        {"id": "c:u-s", "t": "concept", "label": "u.s.", "size": 4},
    ],
    "links": [
        {"source": "s:1", "target": "t:0", "rel": "in_topic"},
        {"source": "t:0", "target": "c:ai", "rel": "topic_concept", "w": 0.4},
        {"source": "t:7", "target": "c:u-s", "rel": "topic_concept"},
        {"source": "c:ai", "target": "c:u-s", "rel": "related"},
    ],
}

RELATIONS = {
    "nodes": [],
    "links": [
        {"source": "t:0", "target": "t:7", "rel": "t_t", "w": 0.886},
        {"source": "t:7", "target": "t:12", "rel": "t_t"},
        {"source": "t:0", "target": "c:ai", "rel": "t_c"},
    ],
}

IDS = np.array(["s:1", "s:2"])
VECTORS = np.array([[0.5, -0.5], [0.25, 1.0]], dtype=np.float16)


def test_vector_literal_is_plain_decimals():
    """A stray 'np.float32(0.1)' repr inside the bracketed text form fails the
    server-side parse and takes the whole insert down with it."""
    literal = pg_loader.vector_literal(np.array([0.1, -0.25, 1.0], dtype=np.float32))
    assert literal[0] == "[" and literal[-1] == "]"
    assert "(" not in literal and " " not in literal
    values = [float(chunk) for chunk in literal[1:-1].split(",")]
    assert values == pytest.approx([0.1, -0.25, 1.0], abs=1e-6)


def test_story_rows_follow_ddl_column_order():
    """The expected order is parsed out of the DDL itself, so a schema edit that
    forgets story_rows is caught here instead of misaligning the insert."""
    block = re.search(r"stories \(\n(.*?)\n\);", pg_loader.DDL, re.DOTALL).group(1)
    ddl_cols = [line.split()[0] for line in block.splitlines()]
    rows = pg_loader.story_rows(NEWS, IDS, VECTORS)
    assert len(rows[0]) == len(ddl_cols)
    named = dict(zip(ddl_cols, rows[0]))
    assert named["id"] == "s:1"
    assert named["title"] == "hello"
    assert named["url"] == "https://example.com/x"
    assert named["date"] == "2026-02-01"
    assert named["points"] == 42
    assert named["tier"] == 2
    assert named["domain"] == "example.com"
    assert named["topic"] == 7
    assert named["embedding"] == "[0.5,-0.5]"


def test_story_rows_null_out_nan_urls():
    rows = pg_loader.story_rows(NEWS, IDS, VECTORS)
    assert rows[1][2] is None  # position 2 is url in the DDL; NaN must land as SQL NULL
    assert rows[1][0] == "s:2"


def test_story_rows_exit_names_the_unembedded_id():
    with pytest.raises(SystemExit) as excinfo:
        pg_loader.story_rows(NEWS, IDS[:1], VECTORS[:1])
    assert "s:2" in str(excinfo.value)
    assert "SAVE_VECTORS=1" in str(excinfo.value)


def test_hybrid_query_binds_every_cli_param():
    for name in ("like", "tier_max", "since", "limit"):  # exactly what main() passes
        assert f"%({name})s" in pg_loader.HYBRID_QUERY


def test_read_js_roundtrip(tmp_path):
    """Bundles are `window.X = {...};` scripts whose payload URLs contain '='
    themselves — only the first one may split the assignment."""
    payload = {"nodes": [{"id": "s:1", "url": "https://example.com/?a=b"}], "links": []}
    path = tmp_path / "bundle.js"
    path.write_text("window.GRAPH_DATA = " + json.dumps(payload) + ";", encoding="utf-8")
    assert neo4j_loader._read_js(path) == payload


def test_node_rows_shapes():
    rows = neo4j_loader.node_rows(NEWS, TOPICS, GRAPH)
    assert set(rows) == {"Story", "Topic", "Concept"}
    assert rows["Story"][0] == {
        "id": "s:1",
        "title": "hello",
        "url": "https://example.com/x",
        "date": "2026-02-01",
        "points": 42,
        "tier": 2,
        "domain": "example.com",
    }
    assert type(rows["Story"][0]["points"]) is int  # numpy int64 breaks driver serialisation
    assert rows["Topic"][0] == {"id": 0, "label": "alpha · beta", "area": "ai", "size": 3}
    assert rows["Concept"] == [{"name": "ai"}, {"name": "u.s."}]


def test_story_nodes_drop_nan_urls():
    rows = neo4j_loader.node_rows(NEWS, TOPICS, GRAPH)
    assert rows["Story"][1]["url"] is None  # a NaN float here would poison the SET clause
    assert rows["Story"][1]["id"] == "s:2"


def test_edge_rows_by_relationship_type():
    rows = neo4j_loader.edge_rows(NEWS, GRAPH, RELATIONS)
    assert set(rows) == {"IN_TOPIC", "ANCHORED_BY", "SIMILAR_TO", "RELATED"}
    assert rows["IN_TOPIC"] == [{"story": "s:1", "topic": 7}, {"story": "s:2", "topic": 0}]
    assert rows["ANCHORED_BY"] == [
        {"topic": 0, "concept": "ai", "weight": 0.4},
        {"topic": 7, "concept": "u.s.", "weight": None},
    ]


def test_similar_to_carries_weights_and_skips_t_c():
    rows = neo4j_loader.edge_rows(NEWS, GRAPH, RELATIONS)
    assert rows["SIMILAR_TO"] == [
        {"a": 0, "b": 7, "weight": 0.886},
        {"a": 7, "b": 12, "weight": None},  # a bundle without "w" still loads
    ]


def test_related_links_use_concept_labels():
    rows = neo4j_loader.edge_rows(NEWS, GRAPH, RELATIONS)
    assert rows["RELATED"] == [{"a": "ai", "b": "u.s."}]  # not the slugged "u-s" id


def test_documented_queries_stay_on_loaded_edges():
    """Every relationship a demo query traverses must be one the loader
    creates, or the query silently returns nothing on a fresh database."""
    assert set(neo4j_loader.DOCUMENTED_QUERIES) == {"concept_hubs", "bridges", "concept_path"}
    for name, cypher in neo4j_loader.DOCUMENTED_QUERIES.items():
        used = set(re.findall(r":([A-Z_]{2,})", cypher))
        assert used and used <= set(neo4j_loader.CYPHER), name
    assert "shortestPath" in neo4j_loader.DOCUMENTED_QUERIES["concept_path"]
    assert "NOT" in neo4j_loader.DOCUMENTED_QUERIES["bridges"]


def test_cypher_batches_all_unwind():
    for name, cypher in neo4j_loader.CYPHER.items():
        assert cypher.startswith("UNWIND $rows AS row"), name  # _run_batched slices $rows
