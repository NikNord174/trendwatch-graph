"""Entry point: page setup and tab dispatch.

One section renders per rerun on purpose: the map tabs inline a multi-megabyte
data bundle each, and a tabs-layout would embed all of them at once.
"""

import streamlit as st

import chat
import data
import maps
import ontology
import theme
import trends

st.set_page_config(page_title="trendwatch-graph", layout="wide")
theme.apply()

st.title("Trendwatch graph")
st.caption(
    "A trend-watching knowledge graph over six months of tech news: stories "
    "clustered into topics, anchored to shared concepts, scored week by week."
)

SECTIONS = ["Trend map", "Timeline", "Topics", "Relations", "Ontology", "Chat"]

# The maps deep-link into the Topics section via ?topic=<id>; a deep-link
# navigation reloads the page, so seeding the initial section here is enough.
if "section" not in st.session_state:
    st.session_state["section"] = "Topics" if "topic" in st.query_params else "Trend map"

selected = st.segmented_control("Section", SECTIONS, key="section", label_visibility="collapsed")

if selected == "Trend map":
    maps.render_trend_map()
elif selected == "Timeline":
    maps.render_timeline()
elif selected == "Topics":
    trends.render(data.load_topics(), data.load_news(), data.load_snapshot_meta())
elif selected == "Relations":
    maps.render_relations()
elif selected == "Ontology":
    ontology.render()
elif selected == "Chat":
    chat.render()
