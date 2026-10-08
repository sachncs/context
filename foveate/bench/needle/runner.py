"""Runs needle cases through the pipelines and reports length x depth grids."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import statistics
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.bench.longdoc import pipelines
from foveate.bench.needle import cases as cases_lib
from foveate.bench.needle import haystack, sources
from foveate.documents import Document
from foveate.internals import concurrency

DEFAULT_PIPELINES = ("full-context", "truncate", "naive-rag", "foveate")


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """How to run.

    Attributes:
        pipelines: Pipeline names to compare.
        budget: Prompt budget for the evidence-limited pipelines.
        concurrency: Cases processed in parallel.
        retrieval: `bm25`, `embedding` or `hybrid` for `naive-rag` and
            `foveate`.
        inference: Allow one inference step (needed for non-literal needles).
    """

    pipelines: tuple[str, ...] = DEFAULT_PIPELINES
    budget: int = 4_000
    concurrency: int = 3
    retrieval: str = "bm25"
    inference: bool = False

    def __post_init__(self) -> None:
        if self.budget < 2_000 or self.concurrency < 1:
            raise errors.ConfigError("need budget >= 2000, concurrency >= 1")
        for name in self.pipelines:
            pipelines.Pipeline.registry.get(name)


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """One case answered by one pipeline."""

    case: cases_lib.Case
    pipeline: str
    correct: bool
    result: pipelines.Result


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    """Aggregate of all outcomes at one (pipeline, family, length, depth)."""

    pipeline: str
    family: str
    length: int
    depth: float
    total: int
    correct: int
    overflow: int
    errors: int
    prompt_tokens: float
    latency_p50: float

    @property
    def accuracy(self) -> float:
        """Returns correct / total."""
        return self.correct / self.total if self.total else 0.0


def build_document(case: cases_lib.Case, source: sources.Source) -> Document:
    """Builds the haystack for `case` with its needles hidden inside."""
    pages = source.pages(case.length, case.seed)
    for sentence, depth in zip(case.sentences, case.depths, strict=True):
        pages = haystack.insert(pages, sentence, depth)
    return haystack.to_document(
        f"haystack_{case.length}", pages, source.tokenizer
    )


@dataclasses.dataclass(frozen=True)
class Report:
    """All outcomes of a needle run."""

    outcomes: tuple[Outcome, ...]
    model: str
    budget: int
    timestamp: str

    def cells(self) -> list[Cell]:
        """Aggregates outcomes per (pipeline, family, length, depth)."""
        groups: dict[tuple[str, str, int, float], list[Outcome]] = defaultdict(
            list
        )
        for o in self.outcomes:
            key = (o.pipeline, o.case.family, o.case.length, o.case.depth)
            groups[key].append(o)
        cells = []
        for (name, family, length, depth), group in sorted(groups.items()):
            ran = [o.result for o in group if not o.result.error]
            cells.append(
                Cell(
                    pipeline=name,
                    family=family,
                    length=length,
                    depth=depth,
                    total=len(group),
                    correct=sum(o.correct for o in group),
                    overflow=sum(o.result.overflow for o in group),
                    errors=sum(
                        bool(o.result.error) and not o.result.overflow
                        for o in group
                    ),
                    prompt_tokens=statistics.fmean(r.prompt_tokens for r in ran)
                    if ran
                    else 0.0,
                    latency_p50=statistics.median(r.seconds for r in ran)
                    if ran
                    else 0.0,
                )
            )
        return cells

    def by_length(self) -> dict[tuple[str, str], dict[int, float]]:
        """Returns accuracy per (pipeline, family) for each length."""
        groups: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
        for o in self.outcomes:
            groups[(o.pipeline, o.case.family, o.case.length)].append(o.correct)
        out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for (name, family, length), flags in groups.items():
            out[(name, family)][length] = sum(flags) / len(flags)
        return out

    def summary(self) -> dict[str, object]:
        """Returns provenance and the cells as plain data."""
        return {
            "model": self.model,
            "budget": self.budget,
            "timestamp": self.timestamp,
            "cells": [
                dataclasses.asdict(c) | {"accuracy": c.accuracy}
                for c in self.cells()
            ],
        }

    def to_markdown(self) -> str:
        """Renders one length x depth table per (pipeline, family)."""
        lines = [
            "# Needle-in-a-haystack",
            "",
            f"- Date (UTC): {self.timestamp}",
            f"- Answer model: `{self.model}`",
            f"- Evidence budget for limited pipelines: {self.budget} tokens",
            "",
        ]
        cells = self.cells()
        for name, family in sorted({(c.pipeline, c.family) for c in cells}):
            mine = [
                c for c in cells if (c.pipeline, c.family) == (name, family)
            ]
            depths = sorted({c.depth for c in mine})
            lengths = sorted({c.length for c in mine})
            lines += [f"### {name}, {family}", ""]
            lines.append(
                "| length \\ depth | "
                + " | ".join(f"{d:.0%}" for d in depths)
                + " |"
            )
            lines.append("|---|" + "---|" * len(depths))
            for length in lengths:
                row = []
                for depth in depths:
                    found = [
                        c
                        for c in mine
                        if c.length == length and c.depth == depth
                    ]
                    row.append(f"{found[0].accuracy:.0%}" if found else "-")
                lines.append(f"| {length:,} | " + " | ".join(row) + " |")
            lines.append("")
        return "\n".join(lines)

    def write(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes `needle.json` and `needle.md`; returns the md path."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(
            json.dumps(self.summary(), indent=1), encoding="utf-8"
        )
        path = directory / "needle.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


async def run(
    cases: Sequence[cases_lib.Case],
    source: sources.Source,
    runtime: runtime_lib.Runtime,
    config: Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> Report:
    """Answers every case with every pipeline.

    Args:
        cases: The needle cases.
        source: Haystack text provider.
        runtime: The answer model.
        config: Pipelines, budget and concurrency.
        progress: Optional callback receiving one line per finished case.
    """
    cfg = config or Config()
    built = {
        name: pipelines.Pipeline.registry.get(name)(
            runtime, cfg.budget, cfg.retrieval, 0, cfg.inference
        )
        for name in cfg.pipelines
    }

    def job(case: cases_lib.Case) -> Callable[[], Awaitable[list[Outcome]]]:
        async def work() -> list[Outcome]:
            document = build_document(case, source)
            outcomes = []
            for name, pipeline in built.items():
                result = await pipeline.answer(case.question, document)
                right = (
                    not result.error
                    and not result.abstained
                    and case.correct(result.text)
                )
                outcomes.append(Outcome(case, name, right, result))
            if progress:
                progress(
                    f"{case.id}: "
                    + ", ".join(
                        f"{o.pipeline}={'ok' if o.correct else 'x'}"
                        for o in outcomes
                    )
                )
            return outcomes

        return work

    batches = await concurrency.gather_bounded(
        [job(c) for c in cases], cfg.concurrency
    )
    return Report(
        outcomes=tuple(o for batch in batches for o in batch),
        model=runtime.model,
        budget=cfg.budget,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    )
