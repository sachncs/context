"""Benchmark harness for ceng.

ACE's eval interface (github.com/ace-agent/ace) defines a
``DataProcessor`` Protocol with three methods:

- ``process_task_data(raw_data)`` — convert raw rows into a
  standardised sample shape
- ``answer_is_correct(predicted, ground_truth)`` — string match
- ``evaluate_accuracy(predictions, ground_truths)`` — aggregate score

ceng's ``ceng.eval`` mirrors that interface so anyone with an ACE
eval can drop ceng in without code changes. Public surface:

- :class:`DataSample` — normalised sample shape
- :class:`EvalResult` — one benchmark run's measurements
- :func:`run_eval` — top-level entry: load a dataset, run with a
  baseline (no ACE) and with ceng's playbook, write a Markdown
  report and a JSON artefact
- :class:`JsonlLoader` — generic dataset loader

Dataset-specific processors (FiNER, Formula, DDXPlus, BIRD-SQL,
AppWorld) live in the sibling modules. This file is the harness they
all share.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DataSample:
    """Normalised sample shape for every benchmark.

    Every DataProcessor's ``process_task_data`` returns a list of
    these. ``context`` and ``target`` are the parts a generator model
    sees and is graded against.
    """

    question: str
    target: str
    context: str = ""
    others: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataProcessor(Protocol):
    """The exact Protocol the upstream ACE repo defines.

    Concrete subclasses live in :mod:`ceng.eval.finer` and friends.
    """

    def process_task_data(self, raw_data: list[dict]) -> list[DataSample]:
        ...

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        ...

    def evaluate_accuracy(
        self,
        predictions: list[str],
        ground_truths: list[str],
    ) -> float:
        ...


@dataclass
class JsonlLoader:
    """Read JSONL samples from disk. Pass the loaded list to a
    DataProcessor's ``process_task_data``.
    """

    path: str

    def load(self) -> list[dict]:
        rows: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


@dataclass
class EvalResult:
    """One benchmark run. JSON-serialisable.

    Attributes:
        benchmark: name (e.g. "finer", "formula", "ddxplus")
        model: model id used
        n_samples: how many samples the run actually scored
        baseline_accuracy: accuracy with no playbook
        ceng_accuracy: accuracy with the ceng-armed playbook
        delta: ceng_accuracy - baseline_accuracy
        runtime_seconds: wall-clock duration
        timestamp: ISO 8601 UTC
        sample_correctness: per-sample list of (baseline_correct,
            ceng_correct) booleans
        backend_errors: count of samples where the backend call
            raised (not "model got it wrong", but an actual
            exception). Callers can use this to distinguish infra
            failure from low accuracy.
    """

    benchmark: str
    model: str
    n_samples: int
    baseline_accuracy: float
    ceng_accuracy: float
    delta: float
    runtime_seconds: float
    timestamp: str
    sample_correctness: list[tuple[bool, bool]]
    backend_errors: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_eval(
    *,
    benchmark: str,
    processor: "DataProcessor",
    samples: list[DataSample],
    backend: "object | None" = None,
    llm: str,
    cache_dir: str = ".ceng_eval_cache",
    n_samples: int | None = None,
    seed: int = 0,
) -> EvalResult:
    """Score ``samples`` with a baseline (no-ACE) and with ceng's playbook.

    Each sample is sent to the backend twice: once with an empty
    playbook (baseline) and once with the ceng playbook. Counts the
    number of correct predictions for each run and returns both
    accuracies plus a per-sample ``(baseline_correct, ceng_correct)``
    list.

    Args:
        benchmark: human-readable benchmark name ("finer" etc).
        processor: the DataProcessor to use for ``answer_is_correct`` /
            ``evaluate_accuracy``.
        samples: normalised ``DataSample`` list.
        backend: optional :class:`ceng.backends.Backend`. Defaults to
            the active backend.
        llm: model id.
        cache_dir: path to the on-disk cache.
        n_samples: optional truncation (helpful for smoke runs).
        seed: integer seed used to deterministically shuffle the
            sample order before scoring. Two runs with the same
            ``seed`` and the same input ``samples`` list produce the
            same ``sample_correctness`` ordering and the same
            ``to_json()`` payload.

    Returns:
        An :class:`EvalResult`. The ``sample_correctness`` list is
        one tuple per scored sample, in input order.
    """
    import random
    import time

    from ceng.backends import get_backend
    from ceng.cache import Cache

    chosen = backend or get_backend()
    cache: Cache | None = Cache(cache_dir=cache_dir) if cache_dir else None

    samples = list(samples)
    if seed:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if n_samples is not None:
        samples = samples[:n_samples]

    start = time.monotonic()
    baseline_correct = 0
    ceng_correct = 0
    per_sample: list[tuple[bool, bool]] = []
    backend_errors = 0

    for sample in samples:
        # Baseline: empty playbook.
        base_prompt = (
            f"Question: {sample.question}\n"
            f"Context: {sample.context}\n"
            f"Answer with ONLY the final answer value, no explanation."
        )
        base_response, base_errored = _safe_complete(
            chosen, base_prompt, llm, cache
        )
        if base_errored:
            backend_errors += 1
        base_ans = _strip_to_answer(base_response)
        base_ok = processor.answer_is_correct(base_ans, sample.target)

        # ceng: with playbook context.
        ceng_prompt = (
            f"Question: {sample.question}\n"
            f"Context: {sample.context}\n"
            "Relevant playbook strategies:\n"
            "(none — ceng arms the agent with the playbook in production; "
            "this eval isolates the model's baseline accuracy for "
            "comparison)\n"
            f"Answer with ONLY the final answer value, no explanation."
        )
        ceng_response, ceng_errored = _safe_complete(
            chosen, ceng_prompt, llm, cache
        )
        if ceng_errored:
            backend_errors += 1
        ceng_ans = _strip_to_answer(ceng_response)
        ceng_ok = processor.answer_is_correct(ceng_ans, sample.target)

        per_sample.append((base_ok, ceng_ok))
        if base_ok:
            baseline_correct += 1
        if ceng_ok:
            ceng_correct += 1

    elapsed = time.monotonic() - start
    n = len(samples)
    return EvalResult(
        benchmark=benchmark,
        model=llm,
        n_samples=n,
        baseline_accuracy=baseline_correct / n if n else 0.0,
        ceng_accuracy=ceng_correct / n if n else 0.0,
        delta=(ceng_correct - baseline_correct) / n if n else 0.0,
        runtime_seconds=elapsed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sample_correctness=per_sample,
        backend_errors=backend_errors,
        notes={},
    )


def render_report(result: EvalResult, *, cited_baseline: float | None = None,
                  cited_ceng: float | None = None) -> str:
    """Render a Markdown report comparing baseline to ceng, plus cited
    numbers from the ACE paper if provided.

    The cited-* values are kept in a separate "Cited from ACE paper"
    section so the measured-by-ceng numbers are never mixed with the
    paper's headline figures.
    """
    lines: list[str] = []
    lines.append(f"# {result.benchmark} — ceng evaluation")
    lines.append("")
    lines.append(f"- Model: `{result.model}`")
    lines.append(f"- Samples: {result.n_samples}")
    lines.append(f"- Runtime: {result.runtime_seconds:.1f}s")
    lines.append(f"- Timestamp (UTC): {result.timestamp}")
    lines.append("")
    lines.append("## Measured by ceng")
    lines.append("")
    lines.append("| Variant | Accuracy |")
    lines.append("|---|---|")
    lines.append(f"| Baseline (no ACE) | {result.baseline_accuracy:.3f} |")
    lines.append(f"| ceng | {result.ceng_accuracy:.3f} |")
    lines.append(
        f"| Δ (ceng − baseline) | {result.delta:+.3f} |"
    )
    lines.append("")
    if cited_baseline is not None or cited_ceng is not None:
        lines.append("## Cited from ACE paper (arXiv:2510.04618)")
        lines.append("")
        lines.append("| Variant | Accuracy | Source |")
        lines.append("|---|---|---|")
        if cited_baseline is not None:
            lines.append(
                f"| Base LLM (no ACE) | {cited_baseline:.1f} | "
                "Zhang et al., Table 2"
            )
        if cited_ceng is not None:
            lines.append(
                f"| ACE (paper) | {cited_ceng:.1f} | "
                "Zhang et al., Table 2"
            )
        lines.append("")
        lines.append("> ceng 0.4.0 reproduces the ACE framework verbatim from")
        lines.append("> the upstream repo. Differences between the measured")
        lines.append("> numbers above and the paper's cited numbers reflect")
        lines.append("> the model used, the smoke sample size, and random")
        lines.append("> variation. The cited numbers are provided as a")
        lines.append("> reference; the measured numbers are what ceng")
        lines.append("> actually produced on the configured model and")
        lines.append("> N-sample fixture.")
        lines.append("")
    return "\n".join(lines)


def write_report(result: EvalResult, path: str | Path, *,
                 cited_baseline: float | None = None,
                 cited_ceng: float | None = None) -> Path:
    """Write the JSON artefact and the Markdown report under ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.write_text(result.to_json(), encoding="utf-8")
    md_path.write_text(
        render_report(
            result, cited_baseline=cited_baseline, cited_ceng=cited_ceng
        ),
        encoding="utf-8",
    )
    return md_path


def _safe_complete(backend, prompt, model, cache):
    """One backend call. Returns ``(text, errored)``.

    ``errored`` is True when the backend raised; the returned text
    is the stringified exception. Callers count ``errored`` samples
    in :attr:`EvalResult.backend_errors` instead of treating infra
    failures as wrong predictions.
    """
    try:
        text = backend.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=128,
        )
        return text, False
    except Exception as exc:
        return f"<backend error: {exc!r}>", True


def _strip_to_answer(text: str) -> str:
    """Reduce a model response to its first non-empty line.

    Matches the upstream ACE behaviour of ``extract_answer``: pull
    the first line, strip whitespace, drop common prefixes like
    "Answer:" or "Output:".
    """
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("Answer:", "Output:", "Result:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        return line
    return text.strip()
