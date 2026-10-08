"""Builds the data behind the site's foveation explorer from the real library.

    python site/scripts/make-fixtures.py

Needs the parsed FinanceBench filings in the foveate cache (see
scripts/build_gold.py). Only page numbers, tiers and token counts are written;
no filing text is published.
"""

import json
import pathlib

from foveate import Foveator, Runtime, errors
from foveate.bench.longdoc import dataset, gold
from foveate.foveation import Tier

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "site/src/data/foveation.json"
BUDGETS = (2000, 4000, 8000, 16000, 32000)
PICKS = (
    ("3M_2018_10K", "fb_capex"),
    ("BOEING_2022_10K", "fb_rates"),
    ("CVSHEALTH_2022_10K", "fb_dividends"),
)
CODES = {Tier.FULL: "F", Tier.CONDENSED: "C", Tier.OUTLINE: "O", Tier.DROPPED: "D"}
SHORT = {
    "fb_capex": "What was 3M's capital expenditure in fiscal 2018?",
    "fb_rates": "What production rate changes is Boeing forecasting for fiscal 2023?",
    "fb_dividends": "Did CVS Health pay dividends to common shareholders in Q2 of fiscal 2022?",
}


def main() -> None:
    bench = dataset.FinanceBench()
    questions = {q.doc_name: q for q in bench.questions()}
    items = gold.read_items(ROOT / "foveate/bench/longdoc/gold/answerable.jsonl")
    runtime = Runtime.without_llm()
    entries = []
    for doc_name, key in PICKS:
        document = bench.document(questions[doc_name], runtime.tokenizer)
        item = next(
            i
            for i in items
            if i.doc_name == doc_name and len(i.gold_pages) == 1
            and i.meta.get("number_on_page") in ("1", "na")
            and SHORT[key].split()[0] in ("What", "Did")
        )
        gold_page = item.gold_pages[0]
        budgets = {}
        for budget in BUDGETS:
            plan = Foveator(runtime, budget=budget).plan(item.question, [document])
            tiers = {p.page: p.tier for p in plan.foveation.pages}
            budgets[str(budget)] = {
                "tiers": "".join(CODES[tiers[n]] for n in document.numbers),
                "prompt_tokens": plan.prompt_tokens,
            }
        entries.append(
            {
                "id": key,
                "question": SHORT[key],
                "filing": doc_name.replace("_", " "),
                "pages": len(document.pages),
                "document_tokens": document.token_count,
                "gold_page": gold_page,
                "budgets": budgets,
            }
        )
        print(key, len(document.pages), gold_page, {b: v["prompt_tokens"] for b, v in budgets.items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"budgets": list(BUDGETS), "entries": entries}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    try:
        main()
    except errors.FoveateError as exc:
        raise SystemExit(f"failed: {exc}")
