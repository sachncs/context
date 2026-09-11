"""Top-level ``ceng.bench`` module.

Public surface:

- :func:`finer`        — run FiNER eval
- :func:`formula`      — run Formula eval
- :func:`ddxplus`      — run DDXPlus eval
- :func:`appworld`     — run AppWorld eval (gated on credentials)
- :func:`smoke`        — run a smoke eval across the public benchmarks

Each function writes a JSON artefact and a Markdown report to
``bench/results/<benchmark>-<timestamp>``.

The CLI entrypoint in ``ceng.bench.__main__`` exposes the same
functions as ``ceng bench <benchmark> --model <name> --limit <n>``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ceng.backends import get_backend
from ceng.eval import run_eval, write_report

DEFAULT_RESULTS_DIR = Path("bench/results")
DEFAULT_CACHE_DIR = ".ceng/cache"


def _check_credentials() -> bool:
    """Hard-fail when no LLM credential is configured.

    Looks for any of the well-known LLM env vars, including local
    backends like Ollama. If none is set, fail with a clear message.
    """
    candidates = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "OLLAMA_HOST",          # local Ollama server
        "CENG_LITELLM_API_BASE", # any litellm-style override
    ]
    for name in candidates:
        if os.environ.get(name):
            return True
    print(
        "ceng.bench: no LLM credentials detected. Set one of:\n"
        "  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY\n"
        "  OPENROUTER_API_KEY, MISTRAL_API_KEY, COHERE_API_KEY\n"
        "  OLLAMA_HOST (for a local Ollama server)\n"
        "and retry."
    )
    return False


def _bench_path(benchmark: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Compute a unique per-run path under bench/results/."""
    results_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    ts = _dt.now(_tz.utc).strftime("%Y%m%d-%H%M%S")
    return results_dir / f"{benchmark}-{ts}"


def _load_finer_samples(limit: int | None) -> list:
    """Return a FiNER fixture sample list."""
    import json as _json

    from ceng.eval.finer import FiNERProcessor

    fixture_path = (
        Path(__file__).parent / "eval" / "fixtures" / "finer.jsonl"
    )
    if fixture_path.exists():
        rows = [
            _json.loads(line)
            for line in fixture_path.read_text().splitlines()
            if line.strip()
        ]
        if limit is not None:
            rows = rows[:limit]
        return FiNERProcessor().process_task_data(rows)
    from ceng.eval import DataSample
    return [
        DataSample(
            question="Sentence: Apple Inc. reported sales.\nTokens: Apple Inc. reported sales\n\nTags (one per line):",
            target="Apple\tB-ORG\nInc.\tI-ORG\nreported\tO\nsales\tO",
            context="",
        )
    ]


def _load_formula_samples(limit: int | None) -> list:
    import json as _json

    from ceng.eval.formula import FormulaProcessor

    fixture_path = (
        Path(__file__).parent / "eval" / "fixtures" / "formula.jsonl"
    )
    if fixture_path.exists():
        rows = [
            _json.loads(line)
            for line in fixture_path.read_text().splitlines()
            if line.strip()
        ]
        if limit is not None:
            rows = rows[:limit]
        return FormulaProcessor().process_task_data(rows)
    from ceng.eval import DataSample
    return [
        DataSample(
            question="Revenue was $1,000,000; expenses $600,000.\n\nQuestion: What is the net income?",
            target="400000",
            context="",
        )
    ]


def _load_ddxplus_samples(limit: int | None) -> list:
    import json as _json

    from ceng.eval.ddxplus import DDXPlusProcessor

    fixture_path = (
        Path(__file__).parent / "eval" / "fixtures" / "ddxplus.jsonl"
    )
    if fixture_path.exists():
        rows = [
            _json.loads(line)
            for line in fixture_path.read_text().splitlines()
            if line.strip()
        ]
        if limit is not None:
            rows = rows[:limit]
        return DDXPlusProcessor().process_task_data(rows)
    from ceng.eval import DataSample
    return [
        DataSample(
            question="A 35-year-old with sudden severe headache and fever.\n\nOptions:\n0. Migraine\n1. Meningitis\n2. Tension headache\n3. Cluster headache\n\nAnswer with the option number only.",
            target="1",
            context="",
        )
    ]


def finer(model: str, limit: int | None = 30, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Run a FiNER eval. Returns the path to the report."""
    if not _check_credentials():
        raise SystemExit(1)
    from ceng.eval.finer import FiNERProcessor

    samples = _load_finer_samples(limit)
    backend = get_backend()
    result = run_eval(
        benchmark="finer",
        processor=FiNERProcessor(),
        samples=samples,
        backend=backend,
        llm=model,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    out = write_report(
        result,
        _bench_path("finer", results_dir),
        cited_baseline=70.7,
        cited_ceng=78.3,
    )
    print(f"finer: baseline={result.baseline_accuracy:.3f} "
          f"ceng={result.ceng_accuracy:.3f} n={result.n_samples} → {out}")
    return out


def formula(model: str, limit: int | None = 30, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Run a Formula eval. Returns the path to the report."""
    if not _check_credentials():
        raise SystemExit(1)
    from ceng.eval.formula import FormulaProcessor

    samples = _load_formula_samples(limit)
    backend = get_backend()
    result = run_eval(
        benchmark="formula",
        processor=FormulaProcessor(),
        samples=samples,
        backend=backend,
        llm=model,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    out = write_report(
        result,
        _bench_path("formula", results_dir),
        cited_baseline=67.5,
        cited_ceng=85.5,
    )
    print(f"formula: baseline={result.baseline_accuracy:.3f} "
          f"ceng={result.ceng_accuracy:.3f} n={result.n_samples} → {out}")
    return out


def ddxplus(model: str, limit: int | None = 30, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Run a DDXPlus eval. Returns the path to the report."""
    if not _check_credentials():
        raise SystemExit(1)
    from ceng.eval.ddxplus import DDXPlusProcessor

    samples = _load_ddxplus_samples(limit)
    backend = get_backend()
    result = run_eval(
        benchmark="ddxplus",
        processor=DDXPlusProcessor(),
        samples=samples,
        backend=backend,
        llm=model,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    out = write_report(
        result,
        _bench_path("ddxplus", results_dir),
        cited_baseline=75.2,
        cited_ceng=90.2,
    )
    print(f"ddxplus: baseline={result.baseline_accuracy:.3f} "
          f"ceng={result.ceng_accuracy:.3f} n={result.n_samples} → {out}")
    return out


def appworld(model: str, limit: int | None = 30, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Run an AppWorld eval. Requires the gated appworld package + ALFW_API_KEY.

    A real implementation is gated on the AppWorld dataset
    registration wall. Until it lands, calling this function raises
    ``NotImplementedError`` so wrapper scripts that gate on exit
    status correctly report failure rather than seeing a misleading
    success.
    """
    from ceng.eval.appworld import require_appworld
    if not require_appworld():
        raise SystemExit(1)
    if not _check_credentials():
        raise SystemExit(1)
    raise NotImplementedError(
        "appworld eval: real implementation gated on the AppWorld "
        "dataset registration wall; see https://appworld.dev for the "
        "API key. ceng ships the data-processor stub and the curated "
        "multi-turn-agent seed playbook; the eval runner itself is "
        "not yet implemented."
    )


def smoke(model: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Path]:
    """Run a smoke eval across all three public benchmarks. Used by
    commit 27's measure-now loop."""
    if not _check_credentials():
        raise SystemExit(1)
    return {
        "finer": finer(model, limit=10, results_dir=results_dir),
        "formula": formula(model, limit=10, results_dir=results_dir),
        "ddxplus": ddxplus(model, limit=10, results_dir=results_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``ceng bench <benchmark> --model <name> --limit <n>``."""
    parser = argparse.ArgumentParser(
        prog="ceng bench",
        description="Run ceng benchmarks end-to-end.",
    )
    parser.add_argument(
        "benchmark",
        choices=("finer", "formula", "ddxplus", "appworld", "smoke", "all"),
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.benchmark == "all":
        smoke(args.model, results_dir=args.results_dir)
        return 0
    if args.benchmark == "smoke":
        smoke(args.model, results_dir=args.results_dir)
        return 0
    fn = {
        "finer": finer,
        "formula": formula,
        "ddxplus": ddxplus,
        "appworld": appworld,
    }[args.benchmark]
    try:
        fn(args.model, limit=args.limit, results_dir=args.results_dir)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
