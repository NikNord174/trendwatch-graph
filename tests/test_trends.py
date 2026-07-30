import pandas as pd

import trends


def test_story_weight_baseline_and_damping():
    points = pd.Series([50, 500, 5000])
    tier = pd.Series([3, 3, 3])
    w = trends.story_weight(points, tier)
    assert w.iloc[0] == 1.0  # 50 points, tier 3 = neutral baseline
    assert abs(w.iloc[1] - 2.0) < 1e-9
    assert abs(w.iloc[2] - 3.0) < 1e-9


def test_story_weight_tier_multiplier():
    points = pd.Series([50, 50])
    w = trends.story_weight(points, pd.Series([1, 4]))
    assert w.iloc[0] > w.iloc[1]


def test_weekly_signal_shape(news_frame):
    weekly = trends.weekly_signal(news_frame)
    assert set(weekly.columns) == {"topic", "week", "weight"}
    assert weekly.groupby(["topic", "week"]).size().max() == 1
    assert (weekly["weight"] > 0).all()


def test_velocity_detects_rising_topic(news_frame):
    weekly = trends.weekly_signal(news_frame)
    vel = trends.velocity(weekly)
    assert vel.loc[2] > 0  # topic 2 only exists in the last weeks
    assert vel.loc[2] > vel.loc[0]  # steady topic must not outrank it


def test_velocity_flat_topic_is_zero_on_short_history(news_frame):
    """With under 2N weeks the windows shrink symmetrically; a flat signal
    must not read as a surge."""
    short = news_frame[(news_frame["topic"] == 0) & (news_frame["date"] < "2026-02-09")]
    weekly = trends.weekly_signal(short)  # five weeks of identical signal
    assert trends.velocity(weekly).loc[0] == 0.0


def test_filter_topics_by_query_and_area(topics_frame):
    assert trends.filter_topics(topics_frame, "spiky", "all")["id"].tolist() == [1]
    assert trends.filter_topics(topics_frame, "", "ai")["id"].tolist() == [1, 2]
    assert trends.filter_topics(topics_frame, "rising", "ai")["id"].tolist() == [2]


def test_sparkline_series_fixed_length(news_frame):
    weekly = trends.weekly_signal(news_frame)
    spark = trends.sparkline_series(weekly)
    lengths = {len(v) for v in spark.values()}
    assert lengths == {weekly["week"].nunique()}
