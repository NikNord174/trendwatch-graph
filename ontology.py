"""The Ontology tab: schema of the graph plus snapshot provenance."""

import streamlit as st

import data

SCHEMA_DOT = """
digraph {{
  bgcolor="transparent";
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
        color="#3a4150", fontcolor="#e6e9ef"];
  edge [fontname="Helvetica", fontsize=10, color="#6b7383", fontcolor="#9aa3b2"];

  story   [label="Story\\n{stories:,} nodes", fillcolor="#1d2633"];
  topic   [label="Topic\\n{topics} nodes", fillcolor="#26203a"];
  concept [label="Concept\\n{concepts} nodes", fillcolor="#203a30"];

  story   -> topic   [label="IN_TOPIC ({in_topic:,})"];
  topic   -> concept [label="ANCHORED_BY ({topic_concept})"];
  concept -> concept [label="RELATED ({related})"];
}}
"""


def render() -> None:
    counts = data.load_graph_counts()
    meta = data.load_snapshot_meta()

    st.subheader("Schema")
    st.caption(
        "Three node types. Stories carry the observable attributes (date, score, "
        "source tier, domain); topics are clusters over story embeddings; concepts "
        "are the recurring terms that anchor topics to each other."
    )
    st.graphviz_chart(
        SCHEMA_DOT.format(
            stories=counts["nodes"].get("story", 0),
            topics=counts["nodes"].get("topic", 0),
            concepts=counts["nodes"].get("concept", 0),
            in_topic=counts["links"].get("in_topic", 0),
            topic_concept=counts["links"].get("topic_concept", 0),
            related=counts["links"].get("related", 0),
        ),
        width="stretch",
    )

    st.subheader("Snapshot and change detection")
    changes = meta.get("changes_vs_previous", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stories", f"{meta.get('stories', 0):,}")
    col2.metric("New since last run", changes.get("new", 0))
    col3.metric("Changed", changes.get("changed", 0))
    col4.metric("Removed", changes.get("removed", 0))
    st.caption(
        f"Generated {meta.get('generated_at', 'n/a')} over "
        f"{meta.get('window', {}).get('from', '')} to {meta.get('window', {}).get('to', '')}. "
        "Every story carries a SHA-256 of its normalized title and URL "
        "(data/hashes.json). Each pipeline run diffs its hash set against the "
        "previous one; the counters above come straight from that diff. Zeros "
        "mean the corpus did not change between the last two runs."
    )
