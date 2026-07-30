# The pinned pandas needs Python >= 3.11; plain `python3` is older on many
# machines (macOS ships 3.9). Override with `make PYTHON=python3.12 install`.
PYTHON ?= python3.11
PY = .venv/bin/python

.PHONY: venv install install-pipeline run test lint format data docker-build docker-run

venv:
	test -d .venv || $(PYTHON) -m venv .venv

install: venv
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-pipeline: install
	$(PY) -m pip install -r requirements-pipeline.txt

run:
	.venv/bin/streamlit run app.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .

data:  # needs `make install-pipeline` once beforehand
	$(PY) -m pipeline.fetch
	$(PY) -m pipeline.run

docker-build:
	docker build -t trendwatch-graph .

docker-run:
	docker run --rm -p 8501:8501 trendwatch-graph
