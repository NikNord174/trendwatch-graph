import pandas as pd
import pytest


@pytest.fixture
def news_frame() -> pd.DataFrame:
    """Three topics over ten weeks; topic 2 only active in the last four."""
    rows = []
    mondays = pd.date_range("2026-01-05", periods=10, freq="7D")
    for week_no, monday in enumerate(mondays):
        rows.append(
            {
                "id": f"s:a{week_no}",
                "topic": 0,
                "date": monday,
                "points": 100,
                "tier": 2,
                "domain": "example.com",
                "title": f"steady story {week_no}",
                "url": "https://example.com/a",
            }
        )
        if week_no % 2 == 0:
            rows.append(
                {
                    "id": f"s:b{week_no}",
                    "topic": 1,
                    "date": monday,
                    "points": 500,
                    "tier": 1,
                    "domain": "example.org",
                    "title": f"spiky story {week_no}",
                    "url": "https://example.org/b",
                }
            )
        if week_no >= 6:
            rows.append(
                {
                    "id": f"s:c{week_no}",
                    "topic": 2,
                    "date": monday,
                    "points": 50,
                    "tier": 3,
                    "domain": "example.net",
                    "title": f"rising story {week_no}",
                    "url": "https://example.net/c",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def topics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 0,
                "label": "steady · things",
                "area": "infra",
                "size": 10,
                "terms": ["steady", "things"],
                "anchors": {"infra": 0.5},
                "report": "r0",
            },
            {
                "id": 1,
                "label": "spiky · news",
                "area": "ai",
                "size": 5,
                "terms": ["spiky", "news"],
                "anchors": {"ai": 0.9},
                "report": "r1",
            },
            {
                "id": 2,
                "label": "rising · topic",
                "area": "ai",
                "size": 4,
                "terms": ["rising", "topic"],
                "anchors": {"ai": 0.4},
                "report": "r2",
            },
        ]
    )
