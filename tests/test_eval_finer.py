"""Tests for ceng.eval.finer.FiNERProcessor.

Uses an embedded 20-sample FiNER-like fixture (no network required)
so the test runs offline. The fixture uses the standard BIO scheme
on a small mix of PER/LOC/ORG/MISC entities drawn from typical
financial-XBRL sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceng.eval import DataProcessor, run_eval, write_report
from ceng.eval.finer import FiNERProcessor, seed_playbook


@pytest.fixture()
def finer_fixture_path(tmp_path) -> Path:
    """20 FiNER-like samples covering PER, LOC, ORG, MISC and O tags."""
    rows = [
        {"id": f"finer-test-{i}", "tokens": tokens, "labels": labels}
        for i, (tokens, labels) in enumerate(_FINER_FIXTURE)
    ]
    path = tmp_path / "finer.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


_FINER_FIXTURE: list[tuple[list[str], list[str]]] = [
    (["Apple", "Inc.", "reported", "record", "sales", "."],
     ["B-ORG", "I-ORG", "O", "O", "O", "O"]),
    (["Tim", "Cook", "is", "the", "CEO", "."],
     ["B-PER", "I-PER", "O", "O", "O", "O"]),
    (["Microsoft", "is", "based", "in", "Redmond", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "O"]),
    (["He", "joined", "Goldman", "Sachs", "in", "2010", "."],
     ["O", "O", "B-ORG", "I-ORG", "O", "O", "O"]),
    (["The", "report", "covers", "Q1", "2024", "earnings", "."],
     ["O", "O", "O", "O", "O", "O", "O"]),
    (["Apple", "and", "Samsung", "compete", "globally", "."],
     ["B-ORG", "O", "B-ORG", "O", "O", "O"]),
    (["London", "is", "in", "the", "UK", "."],
     ["B-LOC", "O", "O", "O", "B-LOC", "O"]),
    (["John", "Smith", "works", "at", "Google", "."],
     ["B-PER", "I-PER", "O", "O", "B-ORG", "O"]),
    (["Berlin", "and", "Paris", "hosted", "the", "summit", "."],
     ["B-LOC", "O", "B-LOC", "O", "O", "O", "O"]),
    (["Tesla", "Motors", "announced", "earnings", "."],
     ["B-ORG", "I-ORG", "O", "O", "O"]),
    (["Maria", "lives", "in", "Madrid", "and", "works", "at", "BBVA", "."],
     ["B-PER", "O", "O", "B-LOC", "O", "O", "O", "B-ORG", "O"]),
    (["Tokyo", "is", "the", "capital", "of", "Japan", "."],
     ["B-LOC", "O", "O", "O", "O", "B-LOC", "O"]),
    (["Meta", "Platforms", "owns", "Facebook", "and", "Instagram", "."],
     ["B-ORG", "I-ORG", "O", "B-ORG", "O", "B-ORG", "O"]),
    (["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"]),
    (["Amazon", "is", "headquartered", "in", "Seattle", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "O"]),
    (["Satya", "Nadella", "leads", "Microsoft", "since", "2014", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"]),
    (["Paris", "is", "a", "city", "in", "France", "."],
     ["B-LOC", "O", "O", "O", "O", "B-LOC", "O"]),
    (["IBM", "is", "headquartered", "in", "Armonk", "New", "York", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"]),
    (["Sundar", "Pichai", "is", "CEO", "of", "Alphabet", "."],
     ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "O"]),
    (["Toyota", "is", "headquartered", "in", "Toyota", "City", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"]),
]


def test_finer_processor_implements_protocol():
    """DataProcessor is the ceng.eval Protocol; FiNERProcessor satisfies it."""
    p = FiNERProcessor()
    assert isinstance(p, DataProcessor)


def test_finer_process_task_data_builds_datasamples(finer_fixture_path):
    p = FiNERProcessor()
    rows = [json.loads(line) for line in finer_fixture_path.read_text().splitlines() if line.strip()]
    samples = p.process_task_data(rows)
    assert len(samples) == 20
    assert samples[0].question.startswith("Sentence:")
    assert "B-ORG" in samples[0].target
    assert samples[0].others["id"] == "finer-test-0"


def test_finer_answer_is_correct_is_exact_match():
    p = FiNERProcessor()
    gold = "B-ORG\nI-ORG\nO\nO"
    assert p.answer_is_correct(gold, gold) is True
    assert p.answer_is_correct("B-ORG\nI-ORG", gold) is False  # missing tags
    assert p.answer_is_correct("B-ORG\nI-MISC\nO\nO", gold) is False  # wrong type
    # token\\ttag format is also accepted (last column is the tag)
    assert (
        p.answer_is_correct("Apple\tB-ORG\nInc.\tI-ORG\nfoo\tO\nbar\tO", gold)
        is True
    )


def test_finer_evaluate_accuracy_returns_fraction():
    p = FiNERProcessor()
    preds = ["B-ORG\nI-ORG", "B-PER\nI-PER", "O"]
    gold =  ["B-ORG\nI-ORG", "B-PER\nO",     "O"]
    assert p.evaluate_accuracy(preds, gold) == pytest.approx(2 / 3)


def test_finer_seed_playbook_is_non_empty():
    """The seed must be non-empty for ceng to differ from a blank start."""
    from ceng.playbook import parse_playbook
    text = seed_playbook()
    assert len(text) > 100
    # All seven ACE sections present in the parsed playbook so the
    # rendered output reads well.
    pb = parse_playbook(text)
    for section in (
        "strategies_and_insights",
        "formulas_and_calculations",
        "code_snippets_and_templates",
        "common_mistakes_to_avoid",
        "problem_solving_heuristics",
        "context_clues_and_indicators",
        "others",
    ):
        assert section in pb.sections_in_order


def test_finer_seed_playbook_loads_into_playbook():
    from ceng.playbook import parse_playbook
    pb = parse_playbook(seed_playbook())
    assert len(pb.bullets) >= 5
    # Every parsed bullet is in one of the seven sections
    for b in pb.bullets.values():
        assert b.section in {
            "strategies_and_insights",
            "formulas_and_calculations",
            "code_snippets_and_templates",
            "common_mistakes_to_avoid",
            "problem_solving_heuristics",
            "context_clues_and_indicators",
            "others",
        }


def test_finer_full_eval_pipeline_with_mock_backend(finer_fixture_path, tmp_path):
    """run_eval with a FakeBackend scores both baseline and ceng-armed runs."""
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def __init__(self, response: str):
            self.response = response

        def complete(self, messages, model, **kw):
            return self.response

    backend = FakeBackend(response="B-ORG\nI-ORG\nO\nO")
    p = FiNERProcessor()
    rows = [json.loads(line) for line in finer_fixture_path.read_text().splitlines() if line.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="finer",
        processor=p,
        samples=samples,
        backend=backend,
        llm="m",
        cache_dir=str(tmp_path / "cache"),
    )
    assert result.n_samples == 20
    # With a single canned response, every sample scores the same; in
    # this case the canned response rarely matches the gold exactly.
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert 0.0 <= result.ceng_accuracy <= 1.0
    # per-sample correctness list is one tuple per sample
    assert len(result.sample_correctness) == 20
    for tup in result.sample_correctness:
        assert isinstance(tup, tuple) and len(tup) == 2


def test_finer_write_report_emits_md_and_json(finer_fixture_path, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def complete(self, messages, model, **kw):
            return "B-ORG\nI-ORG\nO\nO"

    backend = FakeBackend()
    p = FiNERProcessor()
    rows = [json.loads(line) for line in finer_fixture_path.read_text().splitlines() if line.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="finer", processor=p, samples=samples,
        backend=backend, llm="m", cache_dir=str(tmp_path / "cache"),
    )
    out = write_report(
        result,
        tmp_path / "reports" / "finer-2026-07-20",
        cited_baseline=70.7,
        cited_ceng=78.3,
    )
    assert out.exists()
    assert out.suffix == ".md"
    md = out.read_text()
    assert "## Measured by ceng" in md
    assert "## Cited from ACE paper" in md
    assert "70.7" in md
    assert "78.3" in md
    # JSON sibling exists
    json_path = out.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["benchmark"] == "finer"
    assert payload["n_samples"] == 20
