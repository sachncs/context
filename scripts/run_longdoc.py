"""Runs the long-document benchmark (a dev tool).

    python scripts/run_longdoc.py --dry-run          # sizes, no model calls
    python scripts/run_longdoc.py --n 30 --runs 3    # the published setup

The answer and judge models come from the FOVEATE_* environment variables.
Results go to bench/results/ as longdoc.md and longdoc.json.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

from foveate import Runtime, models
from foveate.bench.longdoc import dataset, gold, runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "foveate/bench/longdoc/gold"
NEEDLE_DOCS = 3
MIN_NEEDLE_TOKENS = 40_000


def select(
    n: int, seed: int
) -> tuple[list[gold.GoldItem], list[dataset.Question]]:
    """Picks `n` questions that have paraphrases, with their related items."""
    answerable = gold.read_items(GOLD_DIR / "answerable.jsonl")
    paraphrases = gold.read_items(GOLD_DIR / "paraphrases.jsonl")
    unanswerable = gold.read_items(GOLD_DIR / "unanswerable.jsonl")
    with_paraphrase = {p.meta["source"] for p in paraphrases}
    bench = dataset.FinanceBench()
    pool = [
        q
        for q in bench.questions()
        if q.id in with_paraphrase
        and any(i.meta["source"] == q.id for i in answerable)
    ]
    chosen = dataset.stratified_sample(pool, n, seed)
    ids = {q.id for q in chosen}
    items = [i for i in answerable if i.meta["source"] in ids]
    items += [p for p in paraphrases if p.meta["source"] in ids]
    items += [u for u in unanswerable if u.meta["source"] in ids]
    return items, chosen


def needle_items(corpus: runner.Corpus, window: int) -> list[gold.GoldItem]:
    """Picks needle sweeps on filings long enough to matter but that fit."""
    needles = gold.read_items(GOLD_DIR / "needles.jsonl")
    sizes = {}
    for name in sorted({n.doc_name for n in needles}):
        sizes[name] = corpus.get(name).token_count
    usable = sorted(
        (
            n
            for n, size in sizes.items()
            if MIN_NEEDLE_TOKENS <= size <= window * 0.8
        ),
        key=lambda n: -sizes[n],
    )[:NEEDLE_DOCS]
    return [n for n in needles if n.doc_name in usable]


def main() -> int:
    """Parses arguments and runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--budget", type=int, default=12_000)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pipelines", default=",".join(runner.DEFAULT_PIPELINES)
    )
    parser.add_argument("--out", default=str(ROOT / "bench/results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    items, chosen = select(args.n, args.seed)
    bench = dataset.FinanceBench()
    with Runtime.from_env() as runtime:
        corpus = runner.Corpus(bench, bench.questions(), runtime)
        window = (
            runtime.context_window
            or models.lookup(runtime.model).context_window
        )
        items += needle_items(corpus, window)
        kinds = {k: sum(i.kind == k for i in items) for k in gold.KINDS}
        print(f"{len(chosen)} questions, items by kind: {kinds}")
        if args.dry_run:
            names = sorted({i.doc_name for i in items})
            total = [corpus.get(n).token_count for n in names]
            print(
                f"{len(names)} filings, {sum(total):,} tokens; "
                f"largest {max(total):,}; window {runtime.context_window}"
            )
            return 0
        config = runner.Config(
            pipelines=tuple(args.pipelines.split(",")),
            budget=args.budget,
            runs=args.runs,
            concurrency=args.concurrency,
        )
        report = asyncio.run(
            runner.run(items, corpus, runtime, runtime, config, print)
        )
    print(report.to_markdown())
    print("written:", report.write(pathlib.Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
