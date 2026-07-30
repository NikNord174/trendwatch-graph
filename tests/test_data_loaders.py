"""The app <-> viewer integration contract: the script-stub replacement."""

from pathlib import Path

import data

ROOT = Path(__file__).resolve().parent.parent


def test_load_map_inlines_data(tmp_path, monkeypatch):
    (tmp_path / "v.html").write_text('<html><script src="d.js"></script></html>')
    (tmp_path / "d.js").write_text("window.X = 1;")
    monkeypatch.setattr(data, "ASSETS", tmp_path)
    page = data.load_map("v.html", "d.js")
    assert page == "<html><script>window.X = 1;</script></html>"


def test_load_map_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ASSETS", tmp_path)
    assert data.load_map("absent.html", "absent.js") is None


def test_every_viewer_contains_its_stub_exactly_once():
    for html_name, data_name in [
        ("trend_map.html", "graph_data.js"),
        ("timeline_map.html", "graph_data.js"),
        ("relations_map.html", "relations_data.js"),
    ]:
        html = (ROOT / "assets" / html_name).read_text(encoding="utf-8")
        stub = f'<script src="{data_name}"></script>'
        assert html.count(stub) == 1, html_name
