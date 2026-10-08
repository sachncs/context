"""Offline examples run in the unit suite; the rest are only compiled."""

import pathlib
import py_compile
import runpy
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
OFFLINE = [
    "01_plan_without_a_model.py",
    "03_compress_without_a_model.py",
    "04_agent_memory.py",
    "05_resilient_backend.py",
    "06_evaluate_your_documents.py",
]


def run_example(name: str) -> None:
    sys.path.insert(0, str(EXAMPLES))
    try:
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    finally:
        sys.path.remove(str(EXAMPLES))


def test_plan_example_shows_the_answer_page_in_full(capsys):
    run_example("01_plan_without_a_model.py")
    out = capsys.readouterr().out
    assert "200 pages" in out and "page  40  FULL" in out


def test_compression_example_keeps_the_fact_with_query_aware_methods(capsys):
    run_example("03_compress_without_a_model.py")
    out = capsys.readouterr().out
    assert "query" in out and "fact kept: True" in out
    assert "tool result:" in out


def test_memory_example_recalls_the_fact(capsys):
    run_example("04_agent_memory.py")
    assert "eu-west-1" in capsys.readouterr().out


def test_resilient_example_reports_a_clear_error(capsys):
    run_example("05_resilient_backend.py")
    assert "Error" in capsys.readouterr().out


def test_evaluation_example_builds_all_three_kinds(capsys):
    run_example("06_evaluate_your_documents.py")
    out = capsys.readouterr().out
    assert "answerable" in out and "needle" in out and "26 weeks" in out


def test_model_example_degrades_without_a_model(capsys, monkeypatch):
    monkeypatch.delenv("FOVEATE_MODEL", raising=False)
    run_example("02_ask_with_citations.py")
    assert "FOVEATE_MODEL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "path",
    sorted(str(p.relative_to(EXAMPLES)) for p in EXAMPLES.rglob("*.py")),
)
def test_every_example_compiles(path, tmp_path):
    py_compile.compile(
        str(EXAMPLES / path), cfile=str(tmp_path / "x.pyc"), doraise=True
    )


def test_evolution_example_degrades_without_a_model(capsys, monkeypatch):
    monkeypatch.delenv("FOVEATE_MODEL", raising=False)
    run_example("07_evolve_a_playbook.py")
    assert "FOVEATE_MODEL" in capsys.readouterr().out


def test_every_offline_example_is_listed():
    names = sorted(p.name for p in EXAMPLES.glob("0*.py"))
    assert set(OFFLINE) <= set(names) and "02_ask_with_citations.py" in names
