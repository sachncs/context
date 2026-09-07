"""Tests for ceng.eval.ddxplus.DDXPlusProcessor.

Embedded fixture: 20 small differential-diagnosis problems.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceng.eval import DataProcessor, run_eval, write_report
from ceng.eval.ddxplus import DDXPlusProcessor, seed_playbook


@pytest.fixture()
def ddx_fixture_path(tmp_path) -> Path:
    rows = [
        {"id": f"ddx-test-{i}", "question": q, "options": opts, "answer": a}
        for i, (q, opts, a) in enumerate(_DDX_FIXTURE)
    ]
    path = tmp_path / "ddx.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


_DDX_FIXTURE: list[tuple[str, list[str], str]] = [
    (
        "A 35-year-old woman with sudden severe headache, neck stiffness, and fever.",
        ["Migraine", "Meningitis", "Tension headache", "Cluster headache"],
        "1",
    ),
    (
        "A 60-year-old man with chest pain radiating to the left arm, sweating, and shortness of breath.",
        ["Heartburn", "Myocardial infarction", "Anxiety attack", "Pneumonia"],
        "1",
    ),
    (
        "A 25-year-old with sudden severe abdominal pain, rigid abdomen, and fever.",
        ["Gastritis", "Appendicitis", "Food poisoning", "IBS"],
        "1",
    ),
    (
        "An elderly patient with sudden one-sided weakness, slurred speech, and facial droop.",
        ["Bell's palsy", "Stroke", "Migraine", "Seizure"],
        "1",
    ),
    (
        "A 40-year-old with sudden shortness of breath, sharp chest pain, and history of recent flight.",
        ["Heart attack", "Pulmonary embolism", "Pneumonia", "Asthma"],
        "1",
    ),
    (
        "A 30-year-old with severe abdominal pain on the right side, fever, and nausea.",
        ["Gastritis", "Appendicitis", "Kidney stones", "Food poisoning"],
        "1",
    ),
    (
        "A child with sore throat, fever, and swollen lymph nodes in the neck.",
        ["Common cold", "Strep throat", "Allergies", "Flu"],
        "1",
    ),
    (
        "An adult with sudden severe headache, sensitivity to light, and nausea.",
        ["Tension headache", "Migraine", "Cluster headache", "Meningitis"],
        "1",
    ),
    (
        "A 50-year-old with progressive memory loss, confusion, and difficulty with daily tasks.",
        ["Depression", "Alzheimer's", "Stress", "Vitamin deficiency"],
        "1",
    ),
    (
        "A patient with rapid heartbeat, weight loss, heat intolerance, and anxiety.",
        ["Anxiety", "Hyperthyroidism", "Diabetes", "PMS"],
        "1",
    ),
    (
        "An elderly with shaking hands, slow movements, and balance problems.",
        ["Stroke", "Parkinson's disease", "Alzheimer's", "Arthritis"],
        "1",
    ),
    (
        "A 20-year-old with severe sore throat, fever, and difficulty swallowing.",
        ["Common cold", "Tonsillitis", "Allergies", "Flu"],
        "1",
    ),
    (
        "A patient with sudden high fever, severe headache, and purple rash.",
        ["Common cold", "Meningococcal meningitis", "Flu", "Allergies"],
        "1",
    ),
    (
        "A child with barking cough, hoarseness, and difficulty breathing at night.",
        ["Common cold", "Croup", "Asthma", "Bronchiolitis"],
        "1",
    ),
    (
        "A pregnant woman with high blood pressure, swelling, and protein in urine.",
        ["Normal pregnancy", "Preeclampsia", "Gestational diabetes", "Anemia"],
        "1",
    ),
    (
        "An adult with sudden severe headache, jaw pain, and visual disturbances.",
        ["Migraine", "Temporal arteritis", "Cluster headache", "Tension headache"],
        "1",
    ),
    (
        "A child with high fever, rash that blanches with pressure, and strawberry tongue.",
        ["Measles", "Scarlet fever", "Chickenpox", "Rubella"],
        "1",
    ),
    (
        "A patient with severe abdominal pain in the upper right, fever, and jaundice.",
        ["Gastritis", "Cholecystitis", "Hepatitis", "Pancreatitis"],
        "1",
    ),
    (
        "An elderly with sudden severe back pain, hypotension, and a pulsating abdominal mass.",
        ["Back strain", "Aortic aneurysm", "Kidney stones", "Pancreatitis"],
        "1",
    ),
    (
        "A patient with sudden facial droop on one side, inability to close eye, and loss of taste.",
        ["Stroke", "Bell's palsy", "Migraine", "Trigeminal neuralgia"],
        "1",
    ),
]


def test_ddx_processor_implements_protocol():
    p = DDXPlusProcessor()
    assert isinstance(p, DataProcessor)


def test_ddx_process_task_data(ddx_fixture_path):
    p = DDXPlusProcessor()
    rows = [json.loads(l) for l in ddx_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    assert len(samples) == 20
    assert "Options:" in samples[0].question
    assert "0." in samples[0].question and "1." in samples[0].question
    assert samples[0].target == "1"


def test_ddx_answer_is_correct_accepts_various_formats():
    p = DDXPlusProcessor()
    assert p.answer_is_correct("1", "1") is True
    assert p.answer_is_correct("1.", "1") is True
    assert p.answer_is_correct(" 2 ", "1") is False
    assert p.answer_is_correct("Option 1", "1") is True  # "1" extracted from "Option 1"
    assert p.answer_is_correct("Meningitis is option 1", "1") is True
    assert p.answer_is_correct("garbage", "1") is False


def test_ddx_evaluate_accuracy():
    p = DDXPlusProcessor()
    preds = ["1", "0", "3", "1"]
    gold =  ["1", "1", "3", "0"]
    assert p.evaluate_accuracy(preds, gold) == pytest.approx(0.5)


def test_ddx_seed_playbook_loads():
    from ceng.playbook import parse_playbook
    pb = parse_playbook(seed_playbook())
    assert len(pb.bullets) >= 5
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


def test_ddx_full_eval_pipeline(ddx_fixture_path, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def __init__(self, response):
            self.response = response

        def complete(self, messages, model, **kw):
            return self.response

    # Canned: always answer 1; for each sample either right or wrong
    backend = FakeBackend(response="1")
    p = DDXPlusProcessor()
    rows = [json.loads(l) for l in ddx_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="ddxplus", processor=p, samples=samples,
        backend=backend, llm="m", cache_dir=str(tmp_path / "cache"),
    )
    assert result.n_samples == 20
    # Every sample has answer "1" in fixture so 100% baseline + ceng
    assert result.baseline_accuracy == 1.0
    assert result.ceng_accuracy == 1.0


def test_ddx_write_report(ddx_fixture_path, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class FakeBackend:
        name: str = "fake"

        def complete(self, messages, model, **kw):
            return "1"

    backend = FakeBackend()
    p = DDXPlusProcessor()
    rows = [json.loads(l) for l in ddx_fixture_path.read_text().splitlines() if l.strip()]
    samples = p.process_task_data(rows)
    result = run_eval(
        benchmark="ddxplus", processor=p, samples=samples,
        backend=backend, llm="m", cache_dir=str(tmp_path / "cache"),
    )
    out = write_report(
        result, tmp_path / "reports" / "ddxplus-2026-07-20",
        cited_baseline=75.2, cited_ceng=90.2,
    )
    md = out.read_text()
    assert "## Measured by ceng" in md
    assert "## Cited from ACE paper" in md
    assert "75.2" in md
    assert "90.2" in md
