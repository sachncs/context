"""Retrieval ablation on the long-document gold set (a dev tool).

    python scripts/eval_retrieval.py --embedding-model <model>

For every sampled question and paraphrase it plans the prompt with each
retrieval setting (no answer model is called) and reports how often the page
with the evidence is sent in full, or at least condensed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import sys

from foveate import Foveator, Runtime, errors
from foveate.backends.embeddings import OpenAIEmbedder
from foveate.bench.longdoc import dataset, runner
from foveate.foveation import FoveationConfig, Tier

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_longdoc

ROOT = pathlib.Path(__file__).resolve().parents[1]


def settings(
    runtime: Runtime, embedder: OpenAIEmbedder | None
) -> dict[str, Foveator]:
    """Returns the Foveator variants to compare."""
    plain = Foveator(runtime, budget=12_000, retrieval="bm25")
    out = {"bm25": plain}
    out["bm25+edges"] = dataclasses.replace(
        plain, foveation=FoveationConfig(order="edges")
    )
    out["bm25+expand"] = dataclasses.replace(plain, expand=2)
    if embedder is not None:
        with_vectors = dataclasses.replace(runtime, embedder=embedder)
        out["hybrid"] = Foveator(
            with_vectors, budget=12_000, retrieval="hybrid"
        )
        out["embedding"] = Foveator(
            with_vectors, budget=12_000, retrieval="embedding"
        )
    return out


async def evaluate(
    variants: dict[str, Foveator], corpus: runner.Corpus, items: list
) -> dict[str, dict[str, float]]:
    """Plans every item under every variant and scores gold-page placement."""
    totals = {
        name: {"full": 0, "shown": 0, "n": 0, "tokens": 0} for name in variants
    }
    for item in items:
        document = corpus.get(item.doc_name)
        for name, foveator in variants.items():
            try:
                plan = await foveator.aplan(item.question, [document])
            except errors.FoveateError as exc:
                print(f"skip {item.id} {name}: {exc}", flush=True)
                continue
            tiers = {p.page: p.tier for p in plan.foveation.pages}
            gold = [tiers.get(n) for n in item.gold_pages]
            totals[name]["n"] += 1
            totals[name]["full"] += any(t is Tier.FULL for t in gold)
            totals[name]["shown"] += any(
                t in (Tier.FULL, Tier.CONDENSED) for t in gold
            )
            totals[name]["tokens"] += plan.prompt_tokens
        print(f"done {item.id}", flush=True)
    return {
        name: {
            "queries": t["n"],
            "gold_page_full": t["full"] / max(1, t["n"]),
            "gold_page_shown": t["shown"] / max(1, t["n"]),
            "prompt_tokens": t["tokens"] / max(1, t["n"]),
        }
        for name, t in totals.items()
    }


def main() -> int:
    """Runs the ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument(
        "--variants", default="", help="comma-separated subset to run"
    )
    parser.add_argument("--out", default=str(ROOT / "bench/results/retrieval"))
    args = parser.parse_args()
    items, unused = run_longdoc.select(args.n, 0)
    del unused
    items = [i for i in items if i.kind in ("answerable", "paraphrase")]
    bench = dataset.FinanceBench()
    embedder = None
    if args.embedding_model:
        key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("FOVEATE_BASE_URL")
        embedder = OpenAIEmbedder(
            args.embedding_model,
            base_url=base,
            api_key=key,
            batch_size=32,
            passage_options={"input_type": "passage"},
            query_options={"input_type": "query"},
        )
    with Runtime.from_env() as runtime:
        corpus = runner.Corpus(bench, bench.questions(), runtime)
        variants = settings(runtime, embedder)
        if args.variants:
            keep = set(args.variants.split(","))
            variants = {n: v for n, v in variants.items() if n in keep}
        result = asyncio.run(evaluate(variants, corpus, items))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
