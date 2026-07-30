"""Snapshot loaders and the map-HTML inliner. All file I/O of the app lives here."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"


@st.cache_data
def load_topics() -> pd.DataFrame:
    rows = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


@st.cache_data
def load_news() -> pd.DataFrame:
    df = pd.read_csv(DATA / "news_topics.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_snapshot_meta() -> dict:
    meta = json.loads((DATA / "hashes.json").read_text(encoding="utf-8"))
    meta.pop("hashes", None)  # the full hash map is pipeline-side detail
    return meta


@st.cache_data
def load_graph_counts() -> dict:
    """Node and edge counts by type, for the ontology view."""
    raw = (ASSETS / "graph_data.js").read_text(encoding="utf-8")
    payload = json.loads(raw.partition("=")[2].strip().rstrip(";"))
    nodes: dict[str, int] = {}
    for n in payload["nodes"]:
        nodes[n["t"]] = nodes.get(n["t"], 0) + 1
    links: dict[str, int] = {}
    for link in payload["links"]:
        links[link["rel"]] = links.get(link["rel"], 0) + 1
    return {"nodes": nodes, "links": links}


@st.cache_data
def load_map(html_name: str, data_name: str) -> str | None:
    """Inline a viewer's data bundle so the page can be embedded as one string.

    The viewers in assets/ are standalone files that pull their data through a
    `<script src>` stub; replacing the stub with the file content is the whole
    integration contract between the app and the maps.
    """
    html_path, data_path = ASSETS / html_name, ASSETS / data_name
    if not html_path.exists() or not data_path.exists():
        return None
    html = html_path.read_text(encoding="utf-8")
    stub = f'<script src="{data_name}"></script>'
    if stub not in html:
        return None
    return html.replace(stub, "<script>" + data_path.read_text(encoding="utf-8") + "</script>")
