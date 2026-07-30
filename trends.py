"""Signal model (pure functions) and the Topics tab."""

import numpy as np
import pandas as pd
import streamlit as st

TIER_WEIGHT = {1: 1.25, 2: 1.1, 3: 1.0, 4: 0.8}


def story_weight(points: pd.Series, tier: pd.Series) -> pd.Series:
    """Score = source quality x attention. Log damping keeps one viral story
    from drowning a whole quarter of steady coverage."""
    attention = 1.0 + np.log10(points.clip(lower=50) / 50.0)
    return tier.map(TIER_WEIGHT).fillna(1.0) * attention


def weekly_signal(news: pd.DataFrame) -> pd.DataFrame:
    """Weighted signal per (topic, week), long format."""
    df = news.dropna(subset=["date"]).copy()
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["weight"] = story_weight(df["points"], df["tier"])
    return df.groupby(["topic", "week"], as_index=False)["weight"].sum()


def velocity(weekly: pd.DataFrame, recent_weeks: int = 4) -> pd.Series:
    """Signal of the last N weeks minus the N before, per topic."""
    if weekly.empty:
        return pd.Series(dtype=float)
    weeks = sorted(weekly["week"].unique())
    recent = set(weeks[-recent_weeks:])
    prior = set(weeks[-2 * recent_weeks : -recent_weeks])
    by_topic = weekly.groupby("topic")
    recent_sum = by_topic.apply(lambda g: g.loc[g["week"].isin(recent), "weight"].sum())
    prior_sum = by_topic.apply(lambda g: g.loc[g["week"].isin(prior), "weight"].sum())
    return (recent_sum - prior_sum).round(1)


def filter_topics(topics: pd.DataFrame, query: str, area: str) -> pd.DataFrame:
    out = topics
    if area and area != "all":
        out = out[out["area"] == area]
    if query:
        needle = query.lower()
        mask = out["label"].str.lower().str.contains(needle, regex=False) | out["terms"].apply(
            lambda ts: any(needle in t for t in ts)
        )
        out = out[mask]
    return out


def sparkline_series(weekly: pd.DataFrame) -> dict[int, list[float]]:
    """Fixed-length per-topic series over the full week range, for table sparklines."""
    if weekly.empty:
        return {}
    weeks = sorted(weekly["week"].unique())
    pivot = weekly.pivot_table(
        index="topic", columns="week", values="weight", fill_value=0.0
    ).reindex(columns=weeks, fill_value=0.0)
    return {int(t): [round(v, 1) for v in row] for t, row in pivot.iterrows()}


def render(topics: pd.DataFrame, news: pd.DataFrame, meta: dict) -> None:
    weekly = weekly_signal(news)
    vel = velocity(weekly)
    spark = sparkline_series(weekly)

    table = topics.copy()
    table["velocity"] = table["id"].map(vel).fillna(0.0)
    table["signal"] = table["id"].map(spark).apply(lambda x: x if isinstance(x, list) else [])

    left, mid, right = st.columns([2, 1, 1])
    query = left.text_input("Search topics", placeholder="term or phrase")
    areas = ["all"] + sorted(a for a in topics["area"].unique() if a)
    area = mid.selectbox("Anchor concept", areas)
    sort = right.selectbox("Sort by", ["size", "velocity"])

    view = filter_topics(table, query, area).sort_values(sort, ascending=False)
    changes = meta.get("changes_vs_previous", {})
    st.caption(
        f"{meta.get('stories', 0):,} stories, {meta.get('topics', 0)} topics, "
        f"{meta.get('window', {}).get('from', '')} to {meta.get('window', {}).get('to', '')}"
        f" · vs previous snapshot: {changes.get('new', 0)} new, "
        f"{changes.get('changed', 0)} changed, {changes.get('removed', 0)} removed"
    )
    st.dataframe(
        view[["label", "area", "size", "velocity", "signal"]],
        width="stretch",
        hide_index=True,
        column_config={
            "label": st.column_config.TextColumn("Topic", width="large"),
            "area": st.column_config.TextColumn("Anchor"),
            "size": st.column_config.NumberColumn("Stories"),
            "velocity": st.column_config.NumberColumn(
                "Velocity", help="signal in the last 4 weeks minus the 4 before"
            ),
            "signal": st.column_config.LineChartColumn("Weekly signal"),
        },
    )

    ids = view["id"].tolist()
    if not ids:
        st.info("No topics match the filter.")
        return
    deep_link = st.query_params.get("topic")
    default = (
        int(deep_link) if deep_link and deep_link.isdigit() and int(deep_link) in ids else ids[0]
    )
    labels = dict(zip(topics["id"], topics["label"]))
    topic_id = st.selectbox(
        "Topic detail", ids, index=ids.index(default), format_func=lambda i: labels.get(i, str(i))
    )
    detail(topics, news, weekly, topic_id)


def detail(topics: pd.DataFrame, news: pd.DataFrame, weekly: pd.DataFrame, topic_id: int) -> None:
    row = topics.loc[topics["id"] == topic_id].iloc[0]
    report_col, chart_col = st.columns([1, 1])
    report_col.markdown(row["report"])
    series = weekly[weekly["topic"] == topic_id].set_index("week")["weight"]
    chart_col.bar_chart(series, height=240)

    stories = news[news["topic"] == topic_id].sort_values("points", ascending=False)
    st.dataframe(
        stories[["title", "date", "points", "tier", "domain", "url"]].head(50),
        width="stretch",
        hide_index=True,
        column_config={
            "title": st.column_config.TextColumn("Story", width="large"),
            "date": st.column_config.DateColumn("Date"),
            "points": st.column_config.NumberColumn("Score"),
            "tier": st.column_config.NumberColumn(
                "Tier", help="1 primary, 2 press, 3 blogs, 4 social"
            ),
            "url": st.column_config.LinkColumn("Link"),
        },
    )
