.PHONY: setup test lint format typecheck check build site clean

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,tokenize]"

test:
	$(BIN)/pytest --cov --cov-report=term-missing

lint:
	$(BIN)/ruff check foveate tests examples
	$(BIN)/ruff format --check foveate tests examples

format:
	$(BIN)/ruff check --fix foveate tests examples
	$(BIN)/ruff format foveate tests examples

typecheck:
	$(BIN)/mypy

check: lint typecheck test

build:
	$(BIN)/python -m build
	$(BIN)/twine check dist/*

site:
	cd site && npm ci && npm run build

clean:
	rm -rf build dist .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
