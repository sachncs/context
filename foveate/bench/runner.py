"""Concurrent benchmark runner with explicit arms and honest reporting."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import time
from collections.abc import Awaitable, Callable, Sequence

from foveate import errors
from foveate import runtime as runtime_lib
from foveate import usage as usage_lib
from foveate.bench import base
from foveate.evolution import grading
from foveate.evolution import playbook as playbook_lib
from foveate.internals import concurrency

ANSWER_TOKENS = 128


@dataclasses.dataclass(frozen=True, slots=True)
class Arm:
    """One experimental condition.

    Attributes:
        name: Label in reports.
        playbook: Playbook injected into every prompt; None for baseline.
    """

    name: str
    playbook: playbook_lib.Playbook | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    """Scores of one arm.

    Attributes:
        name: Arm label.
        accuracy: Correct / scored (infrastructure failures excluded).
        correct: Per-sample grade, None where the backend failed.
        backend_errors: Samples lost to backend failures.
        usage: Tokens spent.
    """

    name: str
    accuracy: float
    correct: tuple[bool | None, ...]
    backend_errors: int
    usage: usage_lib.Usage


@dataclasses.dataclass(frozen=True, slots=True)
class BenchResult:
    """A complete benchmark run.

    Attributes:
        benchmark: Benchmark name.
        model: Model identifier.
        samples: Number of samples per arm.
        arms: Results in arm order; the first is the baseline.
        seconds: Wall-clock duration.
        timestamp: ISO 8601 UTC start time.
        cited_baseline: Paper baseline for context (not measured here).
        cited_ace: Paper ACE score for context (not measured here).
    """

    benchmark: str
    model: str
    samples: int
    arms: tuple[ArmResult, ...]
    seconds: float
    timestamp: str
    cited_baseline: float | None = None
    cited_ace: float | None = None

    @property
    def delta(self) -> float:
        """Returns last-arm minus first-arm accuracy."""
        return self.arms[-1].accuracy - self.arms[0].accuracy

    def to_json(self) -> str:
        """Serialises the result."""
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_markdown(self) -> str:
        """Renders a report; measured and cited numbers stay separate."""
        lines = [
            f"# {self.benchmark} - foveate benchmark",
            "",
            f"- Model: `{self.model}`",
            f"- Samples per arm: {self.samples}",
            f"- Runtime: {self.seconds:.1f}s",
            f"- Timestamp (UTC): {self.timestamp}",
            "",
            "## Measured by foveate",
            "",
            "| Arm | Accuracy | Backend errors | Tokens |",
            "|---|---|---|---|",
        ]
        for arm in self.arms:
            lines.append(
                f"| {arm.name} | {arm.accuracy:.3f} | {arm.backend_errors} "
                f"| {arm.usage.total_tokens} |"
            )
        lines.append(f"| delta (last - first) | {self.delta:+.3f} | | |")
        if self.cited_baseline is not None or self.cited_ace is not None:
            lines += [
                "",
                "## Cited from the ACE paper (arXiv:2510.04618, not measured)",
                "",
                f"- Baseline: {self.cited_baseline}",
                f"- ACE: {self.cited_ace}",
                "",
                "Cited numbers come from a different model and the full "
                "dataset; they are context, not a comparison target.",
            ]
        return "\n".join(lines) + "\n"

    def write(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes `<name>-<timestamp>.json` and `.md`; returns the .md path."""
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("+", "").split(".")[0]
        stem = directory / f"{self.benchmark}-{stamp}"
        stem.with_suffix(".json").write_text(self.to_json(), encoding="utf-8")
        markdown = stem.with_suffix(".md")
        markdown.write_text(self.to_markdown(), encoding="utf-8")
        return markdown


@dataclasses.dataclass(frozen=True)
class Runner:
    """Scores a benchmark under several arms, concurrently.

    Attributes:
        runtime: Backend, cache and concurrency settings.
    """

    runtime: runtime_lib.Runtime

    async def score(
        self,
        benchmark: base.Benchmark,
        sample: grading.Sample,
        arm: Arm,
    ) -> tuple[bool | None, usage_lib.Usage]:
        """Returns `(correct, usage)`; correct is None on backend failure."""
        namespace = f"bench:{benchmark.name}:{arm.name}"
        try:
            completion = await self.runtime.complete(
                benchmark.build_messages(sample, arm.playbook),
                source="bench",
                namespace=namespace,
                max_tokens=ANSWER_TOKENS,
            )
        except errors.BackendError:
            return None, usage_lib.Usage()
        answer = benchmark.extract_answer(completion.text)
        return benchmark.is_correct(answer, sample.target), completion.usage

    async def arun(
        self,
        benchmark: base.Benchmark,
        samples: Sequence[grading.Sample],
        arms: Sequence[Arm],
    ) -> BenchResult:
        """Runs every sample under every arm.

        Raises:
            ConfigError: If `samples` or `arms` is empty.
        """
        if not samples or not arms:
            raise errors.ConfigError("need at least one sample and one arm")
        started = time.monotonic()
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        jobs: list[
            Callable[[], Awaitable[tuple[bool | None, usage_lib.Usage]]]
        ] = []

        def job(
            arm: Arm, sample: grading.Sample
        ) -> Callable[[], Awaitable[tuple[bool | None, usage_lib.Usage]]]:
            async def work() -> tuple[bool | None, usage_lib.Usage]:
                return await self.score(benchmark, sample, arm)

            return work

        for arm in arms:
            jobs.extend(job(arm, sample) for sample in samples)
        outcomes = await concurrency.gather_bounded(
            jobs, self.runtime.concurrency
        )
        results = []
        for index, arm in enumerate(arms):
            chunk = outcomes[index * len(samples) : (index + 1) * len(samples)]
            grades = tuple(outcome[0] for outcome in chunk)
            scored = [g for g in grades if g is not None]
            total_usage = usage_lib.Usage()
            for outcome in chunk:
                total_usage = total_usage + outcome[1]
            results.append(
                ArmResult(
                    name=arm.name,
                    accuracy=sum(scored) / len(scored) if scored else 0.0,
                    correct=grades,
                    backend_errors=len(grades) - len(scored),
                    usage=total_usage,
                )
            )
        return BenchResult(
            benchmark=benchmark.name,
            model=self.runtime.model,
            samples=len(samples),
            arms=tuple(results),
            seconds=time.monotonic() - started,
            timestamp=stamp,
            cited_baseline=benchmark.cited_baseline,
            cited_ace=benchmark.cited_ace,
        )
