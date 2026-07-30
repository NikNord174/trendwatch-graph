"""Shared dark theme. Base colors live in .streamlit/config.toml; this file
only adds the accents Streamlit's theming cannot express."""

import streamlit as st

CSS = """
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 1.5rem; max-width: 1400px; }
h1 { font-size: 1.6rem; letter-spacing: .2px; }
a { color: #9fd0ff; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
div[data-testid="stCaptionContainer"] { color: #8b95a5; }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
