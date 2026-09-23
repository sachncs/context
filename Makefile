.PHONY: setup test lint typecheck check build site

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,tokenize]"

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck:
	$(BIN)/mypy src/ceng

check: lint typecheck test

build:
	$(BIN)/python -m build
	$(BIN)/twine check dist/*

site:
	cd site && npm ci && npm run build
