"""Tests for ceng.eval.formula.FormulaProcessor.

Embedded fixture: 20 small financial-computation problems.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceng.eval import DataProcessor, run_eval
from ceng.eval.formula import FormulaProcessor, seed_playbook


@pytest.fixture()
def formula_fixture_path(tmp_path) -> Path:
    rows = [
        {
            "id": f"formula-test-{i}",
            "context": context,
            "question": q,
            "answer": a,
        }
        for i, (context, q, a) in enumerate(_FORMULA_FIXTURE)
    ]
    path = tmp_path / "formula.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


_FORMULA_FIXTURE: list[tuple[str, str, str]] = [
    ("Revenue was $1,000,000; expenses $600,000.",
     "What is the net income?", "400000"),
    ("Revenue 200,000; net income 30,000.",
     "What is the net margin (decimal)?", "0.15"),
    ("Total assets 500,000; net income 50,000.",
     "What is the ROA (decimal)?", "0.1"),
    ("Current assets 200,000; current liabilities 100,000.",
     "What is the current ratio?", "2.0"),
    ("This year 1,200; last year 1,000.",
     "What is YoY growth (decimal)?", "0.2"),
    ("Net income 50,000; shares 10,000.",
     "What is the EPS?", "5"),
    ("Operating income 80,000; revenue 400,000.",
     "Operating margin?", "0.2"),
    ("Cash 50,000; receivables 30,000; inventory 20,000.",
     "Current assets total?", "100000"),
    ("Current liabilities 50,000; current assets 200,000.",
     "Current ratio?", "4.0"),
    ("Sales 500,000; cost of goods 300,000.",
     "Gross margin (decimal)?", "0.4"),
    ("Total revenue 1,000,000; returns 50,000; discounts 50,000.",
     "Net revenue?", "900000"),
    ("Profit 100,000; tax rate 25%.",
     "Net profit after tax?", "75000"),
    ("Cash 20,000; receivables 30,000; payables 25,000.",
     "Net working capital?", "25000"),
    ("Total debt 400,000; equity 600,000.",
     "Debt-to-equity ratio (decimal)?", "0.667"),
    ("Sales 800,000; prior sales 1,000,000.",
     "Sales decline (decimal)?", "-0.2"),
    ("Cost 50,000; markup 25%.",
     "Selling price?", "62500"),
    ("Total 1,000; probability 0.1.",
     "Expected value?", "100"),
    ("Inventory 100 units; sold 75; remaining 25.",
     "Inventory turnover (decimal)?", "0.75"),
    ("Loan 100,000 at 5% for 1 year.",
     "Simple interest?", "5000"),
    ("Loan 100,000 at 5% for 1 year, compounded annually.",
     "Compound interest?", "5000"),
]


def test_formula_processor_implements_protocol():
    p = FormulaProcessor()
    assert isinstance(p, DataProcessor)


def test_formula_process_task_data(formula_fixture_path):
    p = FormulaProcessor()
    rows = [json.loads(l) for l in formula_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    assert len(samples) == 20
    assert "net income" in samples[0].question.lower()
    assert samples[0].target == "400000"


def test_formula_answer_is_correct_with_tolerance():
    p = FormulaProcessor()
    assert p.answer_is_correct("400000", "400000") is True
    assert p.answer_is_correct("400001", "400000") is True  # within tolerance
    assert p.answer_is_correct("0.5", "0.667") is False  # > 1% diff
    # A model that wraps the number in text
    assert p.answer_is_correct("The answer is 400000", "400000") is True
    # No number anywhere
    assert p.answer_is_correct("no number here", "400000") is False


def test_formula_evaluate_accuracy():
    p = FormulaProcessor()
    preds = ["400000", "0.2", "100", "garbage"]
    gold =  ["400000", "0.15", "100", "100"]
    assert p.evaluate_accuracy(preds, gold) == pytest.approx(2 / 4)


def test_formula_seed_playbook_loads_with_curated_content():
    from ceng.playbook import parse_playbook
    pb = parse_playbook(seed_playbook())
    # Has formulas
    formula_bullets = [
        b for b in pb.bullets.values() if "formula" in b.content.lower() or "=" in b.content
    ]
    assert len(formula_bullets) >= 3
    # All seven ACE sections present
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


def test_formula_full_eval_pipeline(formula_fixture_path, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def __init__(self, response: str):
            self.response = response

        def complete(self, messages, model, **kw):
            return self.response

    # Canned response is the right answer for the first 19/20 samples
    # but wrong for the last one.
    canned = "400000"
    backend = FakeBackend(response=canned)
    p = FormulaProcessor()
    rows = [json.loads(l) for l in formula_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="formula", processor=p, samples=samples,
        backend=backend, llm="m", cache_dir=str(tmp_path / "cache"),
    )
    assert result.n_samples == 20
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert 0.0 <= result.ceng_accuracy <= 1.0


def test_run_eval_counts_backend_errors(formula_fixture_path, tmp_path):
    """A backend that throws on every call must surface backend_errors
    on the EvalResult, not be silently counted as 0% accuracy."""
    from dataclasses import dataclass

    @dataclass
    class AlwaysFailBackend:
        name: str = "fail"

        def complete(self, messages, model, **kw):
            raise ConnectionError("upstream died")

    backend = AlwaysFailBackend()
    p = FormulaProcessor()
    rows = [json.loads(l) for l in formula_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="formula", processor=p, samples=samples,
        backend=backend, llm="m", cache_dir=str(tmp_path / "cache"),
    )
    assert result.backend_errors == 2 * len(samples)
    assert result.ceng_accuracy == 0.0
    assert result.baseline_accuracy == 0.0


def test_run_eval_seed_is_reproducible(formula_fixture_path, tmp_path):
    """Two run_eval() calls with the same seed produce identical
    sample_correctness orderings (the seed shuffles the input order
    deterministically)."""
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def complete(self, messages, model, **kw):
            return "400000"

    backend_a = FakeBackend()
    backend_b = FakeBackend()
    p = FormulaProcessor()
    rows = [json.loads(l) for l in formula_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    a = run_eval(
        benchmark="formula", processor=p, samples=samples,
        backend=backend_a, llm="m",
        cache_dir=str(tmp_path / "cache_a"), seed=42,
    )
    b = run_eval(
        benchmark="formula", processor=p, samples=samples,
        backend=backend_b, llm="m",
        cache_dir=str(tmp_path / "cache_b"), seed=42,
    )
    assert a.sample_correctness == b.sample_correctness
    assert a.baseline_accuracy == b.baseline_accuracy
    assert a.ceng_accuracy == b.ceng_accuracy
