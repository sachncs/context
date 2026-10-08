---
title: "Benchmarks"
description: "How Foveate is measured against full context, truncation and plain RAG, and what the first full run found."
---

Status: the first full run (30 FinanceBench questions, 3 runs each, four pipelines) is in
progress. This page states the method now and will hold the numbers, including any
where Foveate loses, as soon as the run finishes. Nothing here is estimated.

## What is compared

All four pipelines use the same answer prompt, JSON format and citation check. Only the
way evidence is chosen and assembled differs.

| Pipeline | Behaviour |
|---|---|
| `full-context` | Sends the whole filing. If it exceeds the window, no call is made and the result counts as an overflow failure. |
| `truncate` | Sends the first pages that fit the budget. |
| `naive-rag` | Fixed-size chunks, BM25 top-k up to the budget, one shot, no retry. |
| `foveate` | Retrieve, foveate, answer, verify quotes, retry once, flag or abstain. |

## Data

* Questions and filings: [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
  (CC BY-NC 4.0), 150 questions over real 10-K, 10-Q and 8-K filings. Filings are
  downloaded at run time and are not redistributed.
* Gold sets built for this project (committed in `foveate/bench/longdoc/gold/`):
  * 112 answerable questions with page-level evidence labels (38 FinanceBench
    questions are skipped because the issuer's PDF is unavailable or the evidence text
    cannot be located).
  * 83 unanswerable questions: a question paired with a filing whose text does not
    contain the answer value.
  * 78 paraphrases of 39 questions, written by the model and used for consistency.
  * 40 needle items: a synthetic sentence inserted at 0, 25, 50, 75 and 100% depth
    of 8 filings (lost-in-the-middle sweep).

## Metrics

Correctness is checked deterministically first (numeric match with scale words such as
"million"), then by a model judge with a rubric. Also reported: citation page precision
and recall against the gold pages, share of answers whose quotes are all verified,
abstention accuracy on unanswerable questions, hallucination rate (answering an
unanswerable question), agreement across paraphrases and runs, tokens, latency.

## Reproduce

```bash
python scripts/build_gold.py              # downloads filings, writes gold sets
python scripts/run_longdoc.py --dry-run   # sizes only, no model calls
python scripts/run_longdoc.py --n 30 --runs 3
```
