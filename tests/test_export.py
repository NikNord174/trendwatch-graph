from pipeline import export


def test_domain_of():
    assert export.domain_of("https://www.example.com/a/b") == "example.com"
    assert export.domain_of("http://blog.example.org/x") == "blog.example.org"
    assert export.domain_of("") == "news.ycombinator.com"


def test_source_tier():
    assert export.source_tier("github.com") == 1
    assert export.source_tier("nasa.gov") == 1
    assert export.source_tier("theverge.com") == 2
    assert export.source_tier("somebody.substack.com") == 4
    assert export.source_tier("personal-blog.io") == 3


def test_importance_thresholds():
    assert export.importance(300) == "High"
    assert export.importance(299) == "Medium"
    assert export.importance(120) == "Medium"
    assert export.importance(119) == "Low"


def test_slug():
    assert export.slug("Open Source") == "open-source"
    assert export.slug("c++ / rust") == "c-rust"
