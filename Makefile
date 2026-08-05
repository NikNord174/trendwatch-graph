# The pinned pandas needs Python >= 3.11; plain `python3` is older on many
# machines (macOS ships 3.9). Override with `make PYTHON=python3.12 install`.
PYTHON ?= python3.11
PY = .venv/bin/python

.PHONY: venv install install-pipeline install-backends run test lint format eval data \
	data-cached backends-up backends-down pg-load neo4j-load docker-build docker-run

venv:
	test -d .venv || $(PYTHON) -m venv .venv

install: venv
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-pipeline: install
	$(PY) -m pip install -r requirements-pipeline.txt

install-backends: install
	$(PY) -m pip install -r requirements-backends.txt

run:
	.venv/bin/streamlit run app.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .

eval:  # retrieval scorecard; never fails the build
	$(PY) -m eval.run

data:  # needs `make install-pipeline` once beforehand
	$(PY) -m pipeline.fetch
	$(PY) -m pipeline.run

data-cached:  # rebuild from the cached raw file without refetching
	$(PY) -m pipeline.run $(if $(SAVE_VECTORS),--save-vectors)

backends-up:  # pg on localhost:5433, neo4j browser on :7474; needs `make install-backends` once
	docker compose -f docker-compose.backends.yml up -d --wait

backends-down:
	docker compose -f docker-compose.backends.yml down

pg-load:  # needs data/vectors.npz — `make data-cached SAVE_VECTORS=1`
	$(PY) -m backends.pg_loader

neo4j-load:
	$(PY) -m backends.neo4j_loader

docker-build:
	docker build -t trendwatch-graph .

docker-run:
	docker run --rm -p 8501:8501 trendwatch-graph
