"""Chat over the snapshot: lexical retrieval picks the relevant topics and
stories, an LLM (optional, key via secrets) writes the answer.

Without a key the tab still works as retrieval-only search, so the public
demo never needs a secret to run.
"""

import math
import os
import re

import pandas as pd
import streamlit as st

import data

TOKEN = re.compile(r"[a-z][a-z0-9+#.-]{1,}")
SYSTEM_PROMPT = (
    "You answer questions about a snapshot of tech-news trends. Use only the "
    "provided context: topic profiles and story titles with dates and scores. "
    "Cite topics by name. If the context does not cover the question, say so."
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@st.cache_data
def build_index() -> tuple[pd.DataFrame, dict[str, float]]:
    """One document per topic: a small `core` field (label + terms) and the
    full bag of its story-title tokens."""
    topics = data.load_topics()
    news = data.load_news()
    titles = news.groupby("topic")["title"].apply(" ".join)
    docs = topics[["id", "label", "report"]].copy()
    docs["core"] = (topics["label"] + " " + topics["terms"].apply(" ".join)).apply(
        lambda t: set(tokenize(t))
    )
    title_tokens = topics["id"].map(titles).fillna("").apply(lambda t: set(tokenize(t)))
    docs["tokens"] = [core | body for core, body in zip(docs["core"], title_tokens)]
    df_counts: dict[str, int] = {}
    for tokens in docs["tokens"]:
        for tok in tokens:
            df_counts[tok] = df_counts.get(tok, 0) + 1
    n_docs = max(len(docs), 1)
    idf = {tok: math.log(n_docs / (1 + df)) + 1 for tok, df in df_counts.items()}
    return docs, idf


def score_topic(q_tokens: set[str], core: set[str], tokens: set[str], idf: dict) -> float:
    """Label/term matches dominate; title-bag matches are damped by document
    size, otherwise the biggest topics match any question through sheer
    vocabulary."""
    core_score = sum(idf.get(t, 0.0) for t in q_tokens & core)
    body_score = sum(idf.get(t, 0.0) for t in q_tokens & (tokens - core))
    return 3.0 * core_score + body_score / math.log2(2 + len(tokens))


def retrieve(question: str, top_n: int = 4) -> pd.DataFrame:
    docs, idf = build_index()
    q_tokens = set(tokenize(question))
    scores = docs.apply(lambda r: score_topic(q_tokens, r["core"], r["tokens"], idf), axis=1)
    hits = docs.assign(score=scores).nlargest(top_n, "score")
    return hits[hits["score"] > 0]


def top_stories(topic_ids: list[int], question: str, limit: int = 8) -> pd.DataFrame:
    news = data.load_news()
    subset = news[news["topic"].isin(topic_ids)]
    q_tokens = set(tokenize(question))
    relevance = subset["title"].apply(lambda t: len(q_tokens & set(tokenize(t))))
    return (
        subset.assign(relevance=relevance)
        .sort_values(["relevance", "points"], ascending=False)
        .head(limit)
    )


def api_key() -> str | None:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("OPENAI_API_KEY")


def llm_answer(question: str, context: str, key: str, history: list[dict]) -> str:
    from openai import OpenAI

    model = "gpt-5-mini"
    try:
        model = st.secrets.get("CHAT_MODEL", model)
    except FileNotFoundError:
        pass
    client = OpenAI(api_key=key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # keep follow-ups coherent without unbounded cost
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def build_context(hits: pd.DataFrame, stories: pd.DataFrame) -> str:
    parts = [f"## Topic: {row['label']}\n{row['report']}" for _, row in hits.iterrows()]
    lines = [
        f"- {s.title} ({s.date.date() if pd.notna(s.date) else 'n/a'}, {s.points} points)"
        for s in stories.itertuples()
    ]
    parts.append("## Matching stories\n" + "\n".join(lines))
    return "\n\n".join(parts)


def render() -> None:
    key = api_key()
    if not key:
        st.info(
            "Retrieval-only mode. Add OPENAI_API_KEY to .streamlit/secrets.toml "
            "(see secrets.toml.example) to get generated answers on top of the "
            "retrieved topics."
        )

    history = st.session_state.setdefault("chat_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about the trends in this snapshot")
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question)
    if hits.empty:
        answer = "Nothing in the snapshot matches that. Try terms from the topic labels."
    else:
        stories = top_stories(hits["id"].tolist(), question)
        context = build_context(hits, stories)
        answer = None
        if key:
            try:
                answer = llm_answer(question, context, key, history[:-1])
            except Exception:  # noqa: BLE001 -- any API failure degrades to retrieval
                st.warning("Answer generation is unavailable; showing retrieval results.")
        if not answer:
            answer = "Closest topics and stories in the snapshot:\n\n" + context
    history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
