"""Runs the needle-in-a-haystack grids (a dev tool).

    python scripts/run_needle.py --lengths 8000,16000,32000 --family literal
    python scripts/run_needle.py --family nonliteral        # downloads NoLiMa

The answer model comes from the FOVEATE_* environment variables. Results go
to results/needle/.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import pathlib
import sys

from foveate import Runtime
from foveate.backends.embeddings import OpenAIEmbedder
from foveate.bench import needle
from foveate.bench.needle import cases
from foveate.tokenizers import base as tokenizer_base

Tokenizer = tokenizer_base.Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_cases(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    source_holder: dict[str, needle.Source],
) -> list[cases.Case]:
    """Returns the cases for the requested family and prepares the source."""
    lengths = [int(n) for n in args.lengths.split(",")]
    depths = [float(d) for d in args.depths.split(",")]
    if args.family == "literal":
        source_holder["source"] = needle.ProseSource(tokenizer)
        return [
            cases.literal(n, d, seed)
            for n in lengths
            for d in depths
            for seed in range(args.seeds)
        ]
    if args.family == "multi":
        source_holder["source"] = needle.ProseSource(tokenizer)
        return [
            cases.multi(n, args.needles, seed, d)
            for n in lengths
            for d in depths
            for seed in range(args.seeds)
        ]
    needles, book = needle.load_nolima(tokenizer)
    source_holder["source"] = book
    tests = list(needles[0]["tests"])[: args.seeds]
    return [
        needle.nonliteral(needles, 0, test, n, d, seed)
        for n in lengths
        for d in depths
        for seed, test in enumerate(tests)
    ]


def with_embedder(runtime: Runtime, model: str) -> Runtime:
    """Returns `runtime` with an embedder when `model` is set."""
    if not model:
        return runtime
    embedder = OpenAIEmbedder(
        model,
        base_url=os.environ.get("FOVEATE_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        passage_options={"input_type": "passage"},
        query_options={"input_type": "query"},
    )
    return dataclasses.replace(runtime, embedder=embedder)


def main() -> int:
    """Parses arguments and runs the grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=cases.FAMILIES, default="literal")
    parser.add_argument("--lengths", default="8000,16000,32000")
    parser.add_argument("--depths", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--needles", type=int, default=3)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument(
        "--retrieval", default="bm25", choices=("bm25", "embedding", "hybrid")
    )
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--pipelines", default="full-context,truncate,naive-rag,foveate"
    )
    parser.add_argument("--out", default=str(ROOT / "results/needle"))
    args = parser.parse_args()
    holder: dict[str, needle.Source] = {}
    with Runtime.from_env() as base_runtime:
        runtime = with_embedder(base_runtime, args.embedding_model)
        grid = build_cases(args, runtime.tokenizer, holder)
        config = needle.Config(
            pipelines=tuple(args.pipelines.split(",")),
            budget=args.budget,
            concurrency=args.concurrency,
            retrieval=args.retrieval,
            inference=args.inference,
        )
        print(f"{len(grid)} cases x {len(config.pipelines)} pipelines")
        report = asyncio.run(
            needle.run(grid, holder["source"], runtime, config, print)
        )
    print(report.to_markdown())
    print("written:", report.write(pathlib.Path(args.out) / args.family))
    return 0


if __name__ == "__main__":
    sys.exit(main())
