"""Map tab renderers. Each embeds one self-contained viewer from assets/."""

import streamlit as st

import data

MAP_HEIGHT = 780


def _embed(html_name: str, data_name: str, caption: str) -> None:
    page = data.load_map(html_name, data_name)
    if page is None:
        st.warning(
            f"Missing {html_name} or {data_name} in assets/ — "
            "run `make data` to rebuild the snapshot."
        )
        return
    st.caption(caption)
    st.iframe(page, height=MAP_HEIGHT)


def render_trend_map() -> None:
    _embed(
        "trend_map.html",
        "graph_data.js",
        "Every story in the snapshot, clustered into topics and anchored to shared "
        "concepts. Click a node for details; stories are hidden until toggled.",
    )


def render_timeline() -> None:
    _embed(
        "timeline_map.html",
        "graph_data.js",
        "The same graph with a period filter: drag the handles to see topics grow "
        "and shrink over time.",
    )


def render_relations() -> None:
    _embed(
        "relations_map.html",
        "relations_data.js",
        "Topics tethered to their anchor concepts, with links between related "
        "topics. Click a topic for its profile and top stories.",
    )
