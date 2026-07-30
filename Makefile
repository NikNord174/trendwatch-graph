PY = .venv/bin/python

.PHONY: venv install install-pipeline run test lint format data docker-build docker-run

venv:
	python3 -m venv .venv

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

data:
	$(PY) -m pipeline.fetch
	$(PY) -m pipeline.run

docker-build:
	docker build -t trendwatch-graph .

docker-run:
	docker run --rm -p 8501:8501 trendwatch-graph
