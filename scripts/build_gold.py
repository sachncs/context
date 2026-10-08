"""Builds the long-document gold sets from FinanceBench (a dev tool).

    python scripts/build_gold.py   # answerable + unanswerable + needles
    python scripts/build_gold.py --paraphrases   # needs a model (FOVEATE_* env)

Downloads each filing once into the foveate cache, parses it, and writes
JSON-lines gold files into foveate/bench/longdoc/gold/. Resumable: parsed
documents are cached, so re-running only does the missing work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
import time

from foveate import Runtime, errors, messages
from foveate.bench.longdoc import dataset, gold
from foveate.internals import jsonout

GOLD_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "foveate/bench/longdoc/gold"
)
NEEDLE_DOCS = 8
UNANSWERABLE_PER_DOC = 1
PARAPHRASE_QUESTIONS = 40


def build_core(seed: int) -> None:
    """Builds answerable, unanswerable and needle sets."""
    bench = dataset.FinanceBench()
    questions = bench.questions()
    documents: dict[str, object] = {}
    answerable: list[gold.GoldItem] = []
    skipped = []
    dead: set[str] = set()
    started = time.monotonic()
    for index, question in enumerate(questions, start=1):
        if question.doc_link in dead:
            skipped.append((question.id, "filing unavailable"))
            continue
        try:
            document = documents.get(question.doc_name) or bench.document(
                question
            )
        except errors.FoveateError as exc:
            dead.add(question.doc_link)
            skipped.append((question.id, str(exc)[:80]))
            print(
                f"[{index}/{len(questions)}] SKIP {question.id}: {exc}",
                flush=True,
            )
            continue
        documents[question.doc_name] = document
        item = gold.answerable_item(question, document)  # type: ignore[arg-type]
        if item is None:
            skipped.append((question.id, "evidence not located"))
        else:
            answerable.append(item)
        print(
            f"[{index}/{len(questions)}] {question.doc_name} "
            f"{'ok p.' + str(item.gold_pages) if item else 'no evidence'} "
            f"({time.monotonic() - started:.0f}s)",
            flush=True,
        )
    gold.write_items(GOLD_DIR / "answerable.jsonl", answerable)

    rng = random.Random(seed)
    unanswerable: list[gold.GoldItem] = []
    pool = list(documents.values())
    by_id = {q.id: q for q in questions}
    for item in answerable:
        question = by_id[item.meta["source"]]
        rng.shuffle(pool)
        made = 0
        for wrong in pool:
            candidate = gold.unanswerable_item(question, wrong)  # type: ignore[arg-type]
            if candidate is not None:
                unanswerable.append(candidate)
                made += 1
            if made >= UNANSWERABLE_PER_DOC:
                break
    gold.write_items(GOLD_DIR / "unanswerable.jsonl", unanswerable)

    needles: list[gold.GoldItem] = []
    chosen = sorted(documents)[:: max(1, len(documents) // NEEDLE_DOCS)][
        :NEEDLE_DOCS
    ]
    for name in chosen:
        needles.extend(gold.needle_items(documents[name], seed))  # type: ignore[arg-type]
    gold.write_items(GOLD_DIR / "needles.jsonl", needles)
    print(
        f"answerable={len(answerable)} unanswerable={len(unanswerable)} "
        f"needles={len(needles)} skipped={len(skipped)}"
    )
    (GOLD_DIR / "skipped.json").write_text(json.dumps(skipped, indent=2))


PARAPHRASE_PROMPT = (
    "Rewrite the question below in two different ways that keep exactly the "
    "same meaning, company, period and requested figure. Do not answer it. "
    'Reply with JSON: {{"paraphrases": ["...", "..."]}}\n\nQuestion: {question}'
)


async def build_paraphrases(runtime: Runtime) -> None:
    """Adds model-written paraphrases of sampled answerable questions."""
    answerable = gold.read_items(GOLD_DIR / "answerable.jsonl")
    by_id = {q.id: q for q in dataset.FinanceBench().questions()}
    chosen = dataset.stratified_sample(
        [by_id[i.meta["source"]] for i in answerable], PARAPHRASE_QUESTIONS
    )
    wanted = {q.id for q in chosen}
    items = [i for i in answerable if i.meta["source"] in wanted]
    made: list[gold.GoldItem] = []
    for item in items:
        reply = await runtime.complete(
            (
                messages.Message(
                    messages.Role.USER,
                    PARAPHRASE_PROMPT.format(question=item.question),
                ),
            ),
            source="gold",
            namespace="paraphrase:v1",
            max_tokens=600,
        )
        try:
            variants = jsonout.extract_json(reply.text).get("paraphrases")
        except errors.ValidationError:
            continue
        for number, text in enumerate(
            variants if isinstance(variants, list) else []
        ):
            made.append(
                gold.GoldItem(
                    id=f"{gold.PARAPHRASE}:{item.meta['source']}:{number}",
                    kind=gold.PARAPHRASE,
                    doc_name=item.doc_name,
                    question=str(text),
                    expected=item.expected,
                    gold_pages=item.gold_pages,
                    meta={
                        "source": item.meta["source"],
                        "original": item.question,
                    },
                )
            )
        print(f"paraphrased {item.id}", flush=True)
    gold.write_items(GOLD_DIR / "paraphrases.jsonl", made)
    print(f"paraphrases={len(made)}")


def main() -> int:
    """Runs the build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paraphrases", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.paraphrases:
        with Runtime.from_env() as runtime:
            asyncio.run(build_paraphrases(runtime))
        return 0
    build_core(args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
