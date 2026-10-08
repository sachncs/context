"""Runs the long-document benchmark and reports it."""

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
from foveate.bench.longdoc import dataset, gold, metrics, pipelines
from foveate.documents import Document
from foveate.internals import concurrency


def pct(value: float) -> str:
    """Formats a fraction as a whole-number percentage."""
    return f"{100 * value:.0f}%"


DEFAULT_PIPELINES = ("full-context", "truncate", "naive-rag", "foveate")


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """How to run.

    Attributes:
        pipelines: Pipeline names to compare.
        budget: Prompt token budget for the evidence-limited pipelines.
        runs: Runs per item and pipeline. Run 0 is deterministic; further
            runs are independent samples at temperature 0.7.
        concurrency: Items processed in parallel.
    """

    pipelines: tuple[str, ...] = DEFAULT_PIPELINES
    budget: int = 12_000
    runs: int = 1
    concurrency: int = 3

    def __post_init__(self) -> None:
        if self.runs < 1 or self.concurrency < 1 or self.budget < 2_000:
            raise errors.ConfigError(
                "need runs >= 1, concurrency >= 1, budget >= 2000"
            )
        for name in self.pipelines:
            pipelines.Pipeline.registry.get(name)


class Corpus:
    """Loads and caches the filings the items refer to."""

    def __init__(
        self,
        bench: dataset.FinanceBench,
        questions: Sequence[dataset.Question],
        runtime: runtime_lib.Runtime,
    ) -> None:
        """Maps filing names to the questions that know their URLs."""
        self.bench = bench
        self.by_doc = {q.doc_name: q for q in questions}
        self.runtime = runtime
        self.cache: dict[str, Document] = {}

    def get(self, doc_name: str) -> Document:
        """Returns the parsed filing (downloaded and parsed on first use)."""
        if doc_name not in self.cache:
            question = self.by_doc[doc_name]
            self.cache[doc_name] = self.bench.document(
                question, self.runtime.tokenizer
            )
        return self.cache[doc_name]


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """Aggregated metrics for one pipeline (see `Report.rows`)."""

    pipeline: str
    answerable: int
    accuracy: float
    accuracy_answered: float
    overflow: float
    errors: float
    false_abstain: float
    grounded: float
    verified_citations: float
    page_hit: float
    page_recall: float
    page_precision: float
    unanswerable: int
    correct_abstain: float
    hallucinated: float
    needle: int
    needle_accuracy: float
    needle_by_depth: dict[str, float]
    reliability: float
    prompt_tokens: float
    latency_p50: float


@dataclasses.dataclass(frozen=True)
class Report:
    """All outcomes of a run plus its provenance.

    Attributes:
        outcomes: Every scored result.
        model: Answer model id.
        judge_model: Judge model id.
        budget: Evidence-limited prompt budget.
        timestamp: ISO 8601 UTC start time.
        documents: Distinct filings used.
    """

    outcomes: tuple[metrics.Outcome, ...]
    model: str
    judge_model: str
    budget: int
    timestamp: str
    documents: int

    def rows(self) -> list[Row]:
        """Aggregates outcomes per pipeline."""
        names = list(dict.fromkeys(o.pipeline for o in self.outcomes))
        return [self.row(name) for name in names]

    def row(self, name: str) -> Row:
        """Aggregates one pipeline's outcomes."""
        mine = [o for o in self.outcomes if o.pipeline == name]
        ans = [
            o for o in mine if o.item.kind in (gold.ANSWERABLE, gold.PARAPHRASE)
        ]
        main = [o for o in ans if o.run == 0 and o.item.kind == gold.ANSWERABLE]
        called = [
            o for o in main if not o.result.overflow and not o.result.error
        ]
        given = [o for o in called if not o.result.abstained]
        unans = [
            o for o in mine if o.item.kind == gold.UNANSWERABLE and o.run == 0
        ]
        needles = [o for o in mine if o.item.kind == gold.NEEDLE and o.run == 0]
        by_depth: dict[str, list[float]] = defaultdict(list)
        for o in needles:
            by_depth[o.item.meta.get("depth", "?")].append(float(o.correct))
        used = [o for o in mine if not o.result.overflow and not o.result.error]
        latencies = [o.result.seconds for o in used]
        cites = sum(o.result.citations for o in given)
        return Row(
            pipeline=name,
            answerable=len(main),
            accuracy=metrics.mean([float(o.correct) for o in main]),
            accuracy_answered=metrics.mean([float(o.correct) for o in called]),
            overflow=metrics.mean([float(o.result.overflow) for o in main]),
            errors=metrics.mean(
                [
                    float(bool(o.result.error) and not o.result.overflow)
                    for o in main
                ]
            ),
            false_abstain=metrics.mean(
                [float(o.result.abstained) for o in called]
            ),
            grounded=metrics.mean([float(o.result.grounded) for o in given]),
            verified_citations=(
                sum(o.result.verified for o in given) / cites if cites else 0.0
            ),
            page_hit=metrics.mean([float(o.recall > 0) for o in given]),
            page_recall=metrics.mean([o.recall for o in given]),
            page_precision=metrics.mean(
                [o.precision for o in given if o.result.cited_pages]
            ),
            unanswerable=len(unans),
            correct_abstain=metrics.mean([float(o.correct) for o in unans]),
            hallucinated=metrics.mean(
                [
                    float(not o.result.abstained and not o.result.error)
                    for o in unans
                ]
            ),
            needle=len(needles),
            needle_accuracy=metrics.mean([float(o.correct) for o in needles]),
            needle_by_depth={
                k: metrics.mean(v) for k, v in sorted(by_depth.items())
            },
            reliability=metrics.agreement(ans),
            prompt_tokens=metrics.mean(
                [float(o.result.prompt_tokens) for o in used]
            ),
            latency_p50=statistics.median(latencies) if latencies else 0.0,
        )

    def to_markdown(self) -> str:
        """Renders the report with every number traceable to this run."""
        rows = self.rows()
        header = [r.pipeline for r in rows]

        def table(title: str, lines: list[tuple[str, list[str]]]) -> list[str]:
            out = [
                f"### {title}",
                "",
                "| metric | " + " | ".join(header) + " |",
            ]
            out.append("|---|" + "---|" * len(header))
            out += [
                f"| {name} | " + " | ".join(values) + " |"
                for name, values in lines
            ]
            return [*out, ""]

        md = [
            "# Long-document benchmark",
            "",
            f"- Date (UTC): {self.timestamp}",
            f"- Answer model: `{self.model}`; judge: `{self.judge_model}`",
            f"- Evidence budget for limited pipelines: {self.budget} tokens",
            f"- Filings used: {self.documents} (FinanceBench, CC BY-NC 4.0)",
            "",
        ]
        md += table(
            "Answer quality (answerable questions)",
            [
                ("questions", [str(r.answerable) for r in rows]),
                ("accuracy (all)", [pct(r.accuracy) for r in rows]),
                (
                    "accuracy (when it could answer)",
                    [pct(r.accuracy_answered) for r in rows],
                ),
                (
                    "could not run (context overflow)",
                    [pct(r.overflow) for r in rows],
                ),
                ("errors", [pct(r.errors) for r in rows]),
                ("false abstentions", [pct(r.false_abstain) for r in rows]),
            ],
        )
        md += table(
            "Grounding (answered questions)",
            [
                (
                    "answers with all citations verified",
                    [pct(r.grounded) for r in rows],
                ),
                (
                    "cited quotes found on the cited page",
                    [pct(r.verified_citations) for r in rows],
                ),
                ("cites a gold evidence page", [pct(r.page_hit) for r in rows]),
                ("gold-page recall", [pct(r.page_recall) for r in rows]),
                ("gold-page precision", [pct(r.page_precision) for r in rows]),
            ],
        )
        md += table(
            "Reliability",
            [
                ("unanswerable questions", [str(r.unanswerable) for r in rows]),
                ("correctly abstains", [pct(r.correct_abstain) for r in rows]),
                (
                    "answers anyway (hallucination)",
                    [pct(r.hallucinated) for r in rows],
                ),
                (
                    "agreement across paraphrases/runs",
                    [pct(r.reliability) for r in rows],
                ),
                (
                    "needle accuracy (all depths)",
                    [pct(r.needle_accuracy) for r in rows],
                ),
            ],
        )
        depths = sorted({d for r in rows for d in r.needle_by_depth})
        if depths:
            md += table(
                "Needle accuracy by depth (0 = start, 1 = end)",
                [
                    (d, [pct(r.needle_by_depth.get(d, 0.0)) for r in rows])
                    for d in depths
                ],
            )
        md += table(
            "Cost and speed (calls that ran)",
            [
                (
                    "mean prompt tokens",
                    [f"{r.prompt_tokens:,.0f}" for r in rows],
                ),
                ("median latency (s)", [f"{r.latency_p50:.1f}" for r in rows]),
            ],
        )
        return "\n".join(md)

    def write(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes `longdoc.json` and `longdoc.md`; returns the md path."""
        directory.mkdir(parents=True, exist_ok=True)
        raw = [
            {
                "item": o.item.id,
                "kind": o.item.kind,
                "pipeline": o.pipeline,
                "run": o.run,
                "correct": o.correct,
                "method": o.method,
                "precision": o.precision,
                "recall": o.recall,
                "result": dataclasses.asdict(o.result),
            }
            for o in self.outcomes
        ]
        (directory / "longdoc.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "judge": self.judge_model,
                    "budget": self.budget,
                    "timestamp": self.timestamp,
                    "outcomes": raw,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        (directory / "summary.json").write_text(
            json.dumps(self.summary(), indent=1), encoding="utf-8"
        )
        path = directory / "longdoc.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def summary(self) -> dict[str, object]:
        """Returns provenance and the per-pipeline rows as plain data."""
        return {
            "model": self.model,
            "judge": self.judge_model,
            "budget": self.budget,
            "timestamp": self.timestamp,
            "documents": self.documents,
            "rows": [dataclasses.asdict(row) for row in self.rows()],
        }

    @classmethod
    def load(cls, path: pathlib.Path, items: Sequence[gold.GoldItem]) -> Report:
        """Reads a `longdoc.json` written by `write`.

        Args:
            path: The JSON file.
            items: Gold items to resolve item ids against.

        Raises:
            ValidationError: If an outcome refers to an unknown item.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {item.id: item for item in items}
        outcomes = []
        for raw in data["outcomes"]:
            if raw["item"] not in by_id:
                raise errors.ValidationError(f"unknown item {raw['item']!r}")
            result = dict(raw["result"])
            result["cited_pages"] = tuple(result["cited_pages"])
            outcomes.append(
                metrics.Outcome(
                    item=by_id[raw["item"]],
                    pipeline=raw["pipeline"],
                    run=raw["run"],
                    result=pipelines.Result(**result),
                    correct=raw["correct"],
                    method=raw["method"],
                    precision=raw["precision"],
                    recall=raw["recall"],
                )
            )
        return cls(
            outcomes=tuple(outcomes),
            model=data["model"],
            judge_model=data["judge"],
            budget=data["budget"],
            timestamp=data["timestamp"],
            documents=len({o.item.doc_name for o in outcomes}),
        )


async def run(
    items: Sequence[gold.GoldItem],
    corpus: Corpus,
    runtime: runtime_lib.Runtime,
    judge_runtime: runtime_lib.Runtime,
    config: Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> Report:
    """Runs every item through every pipeline and scores the results.

    Args:
        items: Gold items to evaluate.
        corpus: Provides the documents.
        runtime: The answer model.
        judge_runtime: The model that grades borderline answers.
        config: Pipelines, budget, runs and concurrency.
        progress: Optional callback receiving one line per finished item.

    Returns:
        The `Report`.
    """
    cfg = config or Config()
    built = {
        name: pipelines.Pipeline.registry.get(name)(runtime, cfg.budget)
        for name in cfg.pipelines
    }

    def job(
        item: gold.GoldItem,
    ) -> Callable[[], Awaitable[list[metrics.Outcome]]]:
        async def work() -> list[metrics.Outcome]:
            try:
                document = corpus.get(item.doc_name)
                if item.kind == gold.NEEDLE:
                    document = gold.with_needle(document, item)
            except errors.FoveateError as exc:
                failed = pipelines.Result(error=str(exc)[:200])
                return [
                    await metrics.score(judge_runtime, item, name, 0, failed)
                    for name in built
                ]
            outcomes = []
            for name, pipeline in built.items():
                for run_index in range(cfg.runs):
                    tag = f"r{run_index}" if run_index else ""
                    result = await pipeline.answer(item.question, document, tag)
                    outcomes.append(
                        await metrics.score(
                            judge_runtime, item, name, run_index, result
                        )
                    )
            if progress:
                progress(
                    f"{item.id}: "
                    + ", ".join(
                        f"{o.pipeline}={'ok' if o.correct else 'x'}"
                        for o in outcomes
                    )
                )
            return outcomes

        return work

    batches = await concurrency.gather_bounded(
        [job(item) for item in items], cfg.concurrency
    )
    outcomes = tuple(o for batch in batches for o in batch)
    return Report(
        outcomes=outcomes,
        model=runtime.model,
        judge_model=judge_runtime.model,
        budget=cfg.budget,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        documents=len({i.doc_name for i in items}),
    )
