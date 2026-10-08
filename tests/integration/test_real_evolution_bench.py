"""ACE evolution and benchmarks with a real model."""

import asyncio
import pathlib

from foveate.bench import Arm, Formula, Runner
from foveate.evolution import Evolver, EvolverConfig, Playbook


def test_evolver_learns_from_real_model_feedback(runtime, tmp_path):
    benchmark = Formula()
    samples = benchmark.load_samples(limit=3)
    checkpoint = pathlib.Path(tmp_path) / "ckpt.json"
    result = Evolver(runtime, EvolverConfig(max_reflector_rounds=1)).evolve(
        benchmark.seed_playbook(),
        samples,
        benchmark,
        epochs=1,
        checkpoint=checkpoint,
    )
    assert len(result.steps) == 3 and checkpoint.exists()
    assert all(
        step.usage.total_tokens > 0 or step.cache_hits for step in result.steps
    )
    assert result.accuracy is not None and 0.0 <= result.accuracy <= 1.0
    assert sum(s.backend_errors for s in result.steps) == 0
    # the evolved playbook round-trips and still renders valid text
    assert Playbook.parse(result.playbook.render()).bullets


def test_playbook_is_really_injected_into_the_prompt(runtime):
    benchmark = Formula()
    samples = benchmark.load_samples(limit=4)
    arms = (Arm("baseline"), Arm("seed", benchmark.seed_playbook()))
    result = asyncio.run(Runner(runtime).arun(benchmark, samples, arms))
    baseline, seeded = result.arms
    assert baseline.backend_errors == seeded.backend_errors == 0
    assert 0.0 <= baseline.accuracy <= 1.0 and 0.0 <= seeded.accuracy <= 1.0
    # Real evidence the seed playbook reached the model: it costs more input.
    assert seeded.usage.prompt_tokens > baseline.usage.prompt_tokens
    assert (
        result.cited_ace == 85.5
        and "Measured by foveate" in result.to_markdown()
    )


def test_benchmark_report_files(runtime, tmp_path):
    benchmark = Formula()
    samples = benchmark.load_samples(limit=2)
    result = asyncio.run(
        Runner(runtime).arun(benchmark, samples, [Arm("only")])
    )
    markdown = result.write(tmp_path / "out")
    assert markdown.exists() and markdown.with_suffix(".json").exists()
