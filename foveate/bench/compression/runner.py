"""Sweeps compression ratios: how much can go before answers break?

Each cell reports more than accuracy, because token savings alone say little
(see the "Beyond Token Savings" study): answer F1 and exact match, how many
planted facts survive (critical and weighted atom recall), agreement with the
answer the full context gives, extra model calls and tokens spent compressing,
latency, and which samples were gained or lost against the uncompressed run.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence

from foveate import Context, Foveator, errors
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.bench import scoring
from foveate.bench.compression import tasks as tasks_lib
from foveate.compression import Budget, Overflow
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.internals import concurrency

RATIOS = (1, 2, 4, 8, 16)
FOVEATE_MIN_BUDGET = 1_500
F1_PASS = 0.5
UNCOMPRESSED = "uncompressed"
METHODS = (
    "truncate",
    "window",
    "extractive",
    "selective",
    "query",
    "ushape-drop",
    "tool_output",
    "clear_tool_results",
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
ONLY = {
    "foveate": ("document",),
    "tool_output": ("tool",),
    "clear_tool_results": ("tool",),
    "window": ("history", "atoms"),
}


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """How to sweep.

    Attributes:
        tasks: Task names.
        methods: Method names (some apply to certain tasks only).
        ratios: Compression ratios above 1; the uncompressed run is always
            included as the reference.
        samples: Seeds per cell.
        tokens: Size of the uncompressed context.
        concurrency: Samples processed in parallel.
    """

    tasks: tuple[str, ...] = ("history", "atoms", "tool", "document", "qa")
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
        if max(self.ratios) < 2:
            raise errors.ConfigError("need at least one ratio above 1")
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
    f1: float
    exact: bool
    original_tokens: int
    final_tokens: int
    compressor_tokens: int
    answer_tokens: int
    forced: bool
    seconds: float
    agreement: float
    atom_recall: float
    weighted_atom_recall: float
    density: float
    mutated: int
    omitted: int
    baseline_correct: bool
    error: str = ""


def applies(task: str, method: str) -> bool:
    """Returns whether `method` makes sense for `task`."""
    allowed = ONLY.get(method)
    return allowed is None or task in allowed


def fits_foveate(tokens: int, ratio: int) -> bool:
    """Returns whether Foveator can work with `tokens // ratio` tokens."""
    return tokens // ratio >= FOVEATE_MIN_BUDGET


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
        options["query"] = sample.question
    out = await context.acompress(
        spec, budget=Budget(budget, Overflow.TRUNCATE), **options
    )
    used = out.report.usage.total_tokens if out.report else 0
    return out, used


async def answer(
    runtime: runtime_lib.Runtime, text: str, question: str
) -> tuple[str, int]:
    """Asks the model about `text`; returns the reply and tokens used."""
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
    return reply.text, reply.usage.total_tokens


def flatten(context: Context) -> str:
    """Renders messages as plain text for the reader."""
    return "\n".join(f"{m.role.value}: {m.content}" for m in context.messages)


def judge(sample: tasks_lib.Sample, reply: str) -> tuple[bool, float, bool]:
    """Scores a reply: (correct, token F1, exact match)."""
    f1 = scoring.best_f1(reply, sample.answers)
    exact = any(scoring.exact_match(reply, g) for g in sample.answers)
    if sample.golds:
        return f1 >= F1_PASS or exact, f1, exact
    return scoring.contains(reply, sample.expected), f1, exact


def failed(
    task: tasks_lib.Task, method: str, ratio: int, seed: int, exc: Exception
) -> Row:
    """Builds the row recorded when a cell raises a Foveate error."""
    return Row(
        task.name, method, ratio, seed, False, 0.0, False, 0, 0, 0, 0,
        False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False,
        f"{type(exc).__name__}: {exc}"[:160],
    )  # fmt: skip


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
    started = time.monotonic()
    try:
        full_text = flatten(Context(sample.messages, runtime))
        base_reply = (await answer(runtime, full_text, sample.question))[0]
        base_ok = judge(sample, base_reply)[0]
        used = 0
        forced = False
        if ratio == 1:
            text, final = full_text, original
            reply, spent = base_reply, 0
        elif method == "foveate":
            document = Document(
                "doc",
                tuple(
                    page_lib.Page(i, t, task.tokenizer.count(t), ())
                    for i, t in enumerate(sample.pages, start=1)
                ),
            )
            result = await Foveator(
                runtime, budget=max(budget, FOVEATE_MIN_BUDGET), max_rounds=1
            ).aask(sample.question, [document])
            text, final = result.text, result.foveation.tokens
            reply, spent = result.text, 0
            used = result.usage.total_tokens
        else:
            context, used = await compress(sample, method, budget, runtime)
            text, final = flatten(context), context.token_count
            forced = bool(context.report and context.report.truncated)
            reply, spent = await answer(runtime, text, sample.question)
        right, f1, exact = judge(sample, reply)
        counts = scoring.taxonomy(sample.atoms, text)
        return Row(
            task=task.name,
            method=method,
            ratio=ratio,
            seed=seed,
            correct=right,
            f1=f1,
            exact=exact,
            original_tokens=original,
            final_tokens=final,
            compressor_tokens=used,
            answer_tokens=spent,
            forced=forced,
            seconds=time.monotonic() - started,
            agreement=scoring.token_f1(reply, base_reply),
            atom_recall=scoring.critical_atom_recall(sample.atoms, text),
            weighted_atom_recall=scoring.weighted_atom_recall(
                sample.atoms, text
            ),
            density=scoring.commitment_density(
                sample.atoms, text, task.tokenizer
            ),
            mutated=counts[scoring.MUTATED],
            omitted=counts[scoring.OMITTED],
            baseline_correct=base_ok,
        )
    except errors.FoveateError as exc:
        return failed(task, method, ratio, seed, exc)


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    """Aggregate of all samples at one (task, method, ratio)."""

    task: str
    method: str
    ratio: int
    n: int
    accuracy: float
    f1: float
    exact: float
    achieved_ratio: float
    compressor_tokens: float
    total_tokens: float
    seconds: float
    agreement: float
    atom_recall: float
    weighted_atom_recall: float
    density: float
    mutated: float
    omitted: float
    gained: int
    lost: int
    forced_truncation: float
    errors: int


def mean(values: Sequence[float]) -> float:
    """Returns the mean, or 0.0 for no values."""
    return statistics.fmean(values) if values else 0.0


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
                    f1=mean([r.f1 for r in ok]),
                    exact=mean([float(r.exact) for r in ok]),
                    achieved_ratio=mean(
                        [r.original_tokens / max(1, r.final_tokens) for r in ok]
                    ),
                    compressor_tokens=mean([r.compressor_tokens for r in ok]),
                    total_tokens=mean(
                        [r.compressor_tokens + r.answer_tokens for r in ok]
                    ),
                    seconds=mean([r.seconds for r in ok]),
                    agreement=mean([r.agreement for r in ok]),
                    atom_recall=mean([r.atom_recall for r in ok]),
                    weighted_atom_recall=mean(
                        [r.weighted_atom_recall for r in ok]
                    ),
                    density=mean([r.density for r in ok]),
                    mutated=mean([float(r.mutated) for r in ok]),
                    omitted=mean([float(r.omitted) for r in ok]),
                    gained=sum(
                        r.correct and not r.baseline_correct for r in rows
                    ),
                    lost=sum(
                        r.baseline_correct and not r.correct for r in rows
                    ),
                    forced_truncation=sum(r.forced for r in rows) / len(rows),
                    errors=sum(bool(r.error) for r in rows),
                )
            )
        return out

    def table(
        self,
        cells: Sequence[Cell],
        title: str,
        value: Callable[[Cell], str],
    ) -> list[str]:
        """Renders one methods x ratios table."""
        ratios = sorted({c.ratio for c in cells})
        lines = [f"#### {title}", ""]
        lines.append("| method | " + " | ".join(f"{r}x" for r in ratios) + " |")
        lines.append("|---|" + "---|" * len(ratios))
        for method in sorted({c.method for c in cells}):
            row = []
            for ratio in ratios:
                found = [
                    c for c in cells if c.method == method and c.ratio == ratio
                ]
                row.append(value(found[0]) if found else "-")
            lines.append(f"| {method} | " + " | ".join(row) + " |")
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        """Renders the sweep: accuracy first, then the other measures."""
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
            lines += [f"### {task}", ""]
            lines += self.table(mine, "Correct", lambda c: f"{c.accuracy:.0%}")
            if task == "qa":
                lines += self.table(mine, "F1", lambda c: f"{c.f1:.2f}")
                lines += self.table(
                    mine, "Exact match", lambda c: f"{c.exact:.0%}"
                )
            if any(c.atom_recall or c.omitted or c.mutated for c in mine):
                lines += self.table(
                    mine,
                    "Facts kept (critical atom recall)",
                    lambda c: f"{c.atom_recall:.0%}",
                )
                lines += self.table(
                    mine,
                    "Facts kept, weighted by importance",
                    lambda c: f"{c.weighted_atom_recall:.0%}",
                )
            lines += self.table(
                mine,
                "Agreement with the full-context answer (token F1)",
                lambda c: f"{c.agreement:.2f}",
            )
            lines += self.table(
                mine,
                "Mean tokens spent compressing and answering",
                lambda c: f"{c.total_tokens:,.0f}",
            )
            lines += self.table(
                mine,
                "Mean seconds per sample",
                lambda c: f"{c.seconds:.1f}",
            )
            lines += self.table(
                mine,
                "Samples gained / lost against no compression",
                lambda c: f"+{c.gained} / -{c.lost}",
            )
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
        (task, UNCOMPRESSED, 1, seed)
        for task in cfg.tasks
        for seed in range(cfg.samples)
    ] + [
        (task, method, ratio, seed)
        for task in cfg.tasks
        for method in cfg.methods
        if applies(task, method)
        for ratio in cfg.ratios
        if ratio > 1
        and not (method == "foveate" and not fits_foveate(cfg.tokens, ratio))
        for seed in range(cfg.samples)
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
