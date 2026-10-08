"""Sweeps compression ratios: how much can go before answers break?"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import statistics
from collections import defaultdict
from collections.abc import Awaitable, Callable

from foveate import Context, Foveator, errors
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.bench.compression import tasks as tasks_lib
from foveate.compression import Budget, Overflow
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.internals import concurrency

RATIOS = (1, 2, 4, 8, 16)
FOVEATE_MIN_BUDGET = 1_500
METHODS = (
    "truncate",
    "window",
    "extractive",
    "query",
    "ushape-drop",
    "tool_output",
    "ppa",
    "foveate",
)
ANSWER = (
    "Answer the question using only the context. "
    "Reply with the answer only.\n\n"
    "<context>\n{context}\n</context>\n\nQuestion: {question}"
)
OPTIONS: dict[str, dict[str, object]] = {
    "truncate": {"keep": "head"},
    "ushape-drop": {"head": 1, "tail": 2, "middle": "drop"},
}
SPECS = {"ushape-drop": "ushape"}


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """How to sweep.

    Attributes:
        tasks: Task names.
        methods: Method names (`foveate` applies to the document task only).
        ratios: Compression ratios; 1 means no compression.
        samples: Seeds per cell.
        tokens: Size of the uncompressed context.
        concurrency: Samples processed in parallel.
    """

    tasks: tuple[str, ...] = ("history", "tool", "document")
    methods: tuple[str, ...] = METHODS
    ratios: tuple[int, ...] = RATIOS
    samples: int = 6
    tokens: int = 6_000
    concurrency: int = 3

    def __post_init__(self) -> None:
        if self.samples < 1 or self.concurrency < 1 or self.tokens < 1_000:
            raise errors.ConfigError(
                "need samples, concurrency >= 1 and tokens >= 1000"
            )
        if not self.ratios or min(self.ratios) < 1:
            raise errors.ConfigError("ratios must be >= 1")
        for name in self.tasks:
            tasks_lib.Task.registry.get(name)
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise errors.ConfigError(f"unknown methods: {sorted(unknown)}")


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One sample answered after compression."""

    task: str
    method: str
    ratio: int
    seed: int
    correct: bool
    original_tokens: int
    final_tokens: int
    compressor_tokens: int
    forced: bool
    error: str = ""


def applies(task: str, method: str) -> bool:
    """Returns whether `method` makes sense for `task`."""
    if method == "foveate":
        return task == "document"
    if method == "tool_output":
        return task == "tool"
    return not (method == "window" and task != "history")


async def compress(
    sample: tasks_lib.Sample,
    method: str,
    budget: int,
    runtime: runtime_lib.Runtime,
) -> tuple[Context, int]:
    """Compresses the sample's messages; returns the result and LLM tokens."""
    context = Context(sample.messages, runtime)
    spec = SPECS.get(method, method)
    options = dict(OPTIONS.get(method, {}))
    if method == "query":
        options["query"] = sample.query
    out = await context.acompress(
        spec, budget=Budget(budget, Overflow.TRUNCATE), **options
    )
    used = out.report.usage.total_tokens if out.report else 0
    return out, used


async def answer(runtime: runtime_lib.Runtime, text: str, question: str) -> str:
    """Asks the model about `text`."""
    reply = await runtime.complete(
        (
            messages_lib.Message(
                messages_lib.Role.USER,
                ANSWER.format(context=text, question=question),
            ),
        ),
        source="compression-bench",
        namespace="compression-bench:v1",
        max_tokens=300,
    )
    return reply.text


def flatten(context: Context) -> str:
    """Renders messages as plain text for the reader."""
    return "\n".join(f"{m.role.value}: {m.content}" for m in context.messages)


async def one(
    runtime: runtime_lib.Runtime,
    task: tasks_lib.Task,
    method: str,
    ratio: int,
    seed: int,
) -> Row:
    """Runs one (task, method, ratio, seed) cell."""
    sample = task.build(seed)
    original = sum(task.tokenizer.count(m.content) for m in sample.messages)
    budget = max(64, original // ratio)
    try:
        if ratio == 1:
            context, used = Context(sample.messages, runtime), 0
            text = flatten(context)
            final, forced = original, False
        elif method == "foveate":
            document = Document(
                "doc",
                tuple(
                    page_lib.Page(i, t, task.tokenizer.count(t), ())
                    for i, t in enumerate(sample.pages, start=1)
                ),
            )
            result = await Foveator(
                runtime, budget=max(budget, 1024), max_rounds=1
            ).aask(sample.question, [document])
            right = sample.expected.casefold() in result.text.casefold()
            return Row(
                task.name,
                method,
                ratio,
                seed,
                right,
                original,
                result.foveation.tokens,
                result.usage.total_tokens,
                False,
            )
        else:
            context, used = await compress(sample, method, budget, runtime)
            text = flatten(context)
            final = context.token_count
            forced = bool(context.report and context.report.truncated)
        reply = await answer(runtime, text, sample.question)
        right = sample.expected.casefold() in reply.casefold()
        return Row(
            task.name, method, ratio, seed, right, original, final, used, forced
        )
    except errors.FoveateError as exc:
        return Row(
            task.name,
            method,
            ratio,
            seed,
            False,
            original,
            0,
            0,
            False,
            f"{type(exc).__name__}: {exc}"[:160],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    """Aggregate of all samples at one (task, method, ratio)."""

    task: str
    method: str
    ratio: int
    n: int
    accuracy: float
    achieved_ratio: float
    compressor_tokens: float
    forced_truncation: float
    errors: int


@dataclasses.dataclass(frozen=True)
class Report:
    """All rows of a sweep plus provenance."""

    rows: tuple[Row, ...]
    model: str
    timestamp: str

    def cells(self) -> list[Cell]:
        """Aggregates rows per (task, method, ratio)."""
        groups: dict[tuple[str, str, int], list[Row]] = defaultdict(list)
        for r in self.rows:
            groups[(r.task, r.method, r.ratio)].append(r)
        out = []
        for (task, method, ratio), rows in sorted(groups.items()):
            ok = [r for r in rows if not r.error]
            out.append(
                Cell(
                    task=task,
                    method=method,
                    ratio=ratio,
                    n=len(rows),
                    accuracy=sum(r.correct for r in rows) / len(rows),
                    achieved_ratio=statistics.fmean(
                        r.original_tokens / max(1, r.final_tokens) for r in ok
                    )
                    if ok
                    else 0.0,
                    compressor_tokens=statistics.fmean(
                        r.compressor_tokens for r in ok
                    )
                    if ok
                    else 0.0,
                    forced_truncation=sum(r.forced for r in rows) / len(rows),
                    errors=sum(bool(r.error) for r in rows),
                )
            )
        return out

    def to_markdown(self) -> str:
        """Renders accuracy by method and ratio for each task."""
        cells = self.cells()
        lines = [
            "# Compression sweep",
            "",
            f"- Date (UTC): {self.timestamp}",
            f"- Answer model: `{self.model}`",
            "",
        ]
        for task in sorted({c.task for c in cells}):
            mine = [c for c in cells if c.task == task]
            ratios = sorted({c.ratio for c in mine})
            lines += [f"### {task}: accuracy by compression ratio", ""]
            lines.append(
                "| method | " + " | ".join(f"{r}x" for r in ratios) + " |"
            )
            lines.append("|---|" + "---|" * len(ratios))
            for method in sorted({c.method for c in mine}):
                row = []
                for ratio in ratios:
                    found = [
                        c
                        for c in mine
                        if c.method == method and c.ratio == ratio
                    ]
                    row.append(f"{found[0].accuracy:.0%}" if found else "-")
                lines.append(f"| {method} | " + " | ".join(row) + " |")
            lines.append("")
        return "\n".join(lines)

    def write(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes `summary.json` and `compression.md`; returns the md path."""
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": self.model,
            "timestamp": self.timestamp,
            "cells": [dataclasses.asdict(c) for c in self.cells()],
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )
        path = directory / "compression.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


async def run(
    runtime: runtime_lib.Runtime,
    config: Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> Report:
    """Runs the whole sweep with the answer model `runtime`."""
    cfg = config or Config()
    built = {
        name: tasks_lib.Task.registry.get(name)(runtime.tokenizer, cfg.tokens)
        for name in cfg.tasks
    }
    cells = [
        (task, method, ratio, seed)
        for task in cfg.tasks
        for method in cfg.methods
        if applies(task, method)
        for ratio in cfg.ratios
        for seed in range(cfg.samples)
        if not (ratio == 1 and method != cfg.methods[0])
        and not (
            method == "foveate" and cfg.tokens // ratio < FOVEATE_MIN_BUDGET
        )
    ]

    def job(cell: tuple[str, str, int, int]) -> Callable[[], Awaitable[Row]]:
        async def work() -> Row:
            task, method, ratio, seed = cell
            row = await one(runtime, built[task], method, ratio, seed)
            if progress:
                verdict = "ok" if row.correct else "x"
                progress(f"{task} {method} {ratio}x #{seed}: {verdict}")
            return row

        return work

    rows = await concurrency.gather_bounded(
        [job(c) for c in cells], cfg.concurrency
    )
    return Report(
        rows=tuple(rows),
        model=runtime.model,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    )
