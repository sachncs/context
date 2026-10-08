"""LLM-free examples run in the unit suite; the rest in tests/integration."""

import pathlib
import runpy
import sys

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def run_example(name: str) -> None:
    sys.path.insert(0, str(EXAMPLES))
    try:
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    finally:
        sys.path.remove(str(EXAMPLES))


def test_llm_free_example_runs_without_any_provider(capsys):
    run_example("02_llm_free.py")
    out = capsys.readouterr().out
    assert "extractive" in out and "offloaded" in out


def test_long_document_example_plans_without_a_model(capsys, monkeypatch):
    monkeypatch.delenv("FOVEATE_MODEL", raising=False)
    run_example("08_long_document.py")
    out = capsys.readouterr().out
    assert "200 pages" in out and "page  40  FULL" in out


def test_every_example_is_listed():
    names = sorted(p.name for p in EXAMPLES.glob("0*.py"))
    assert names == [
        "01_quickstart.py",
        "02_llm_free.py",
        "03_fallback_and_persistence.py",
        "04_verify_and_evolve.py",
        "05_pydantic_ai.py",
        "06_google_adk.py",
        "07_langgraph.py",
        "08_long_document.py",
    ]
