---
title: "Evaluating your own pipeline"
description: "Compare full context, truncation, naive RAG and Foveate on your own documents."
---

The benchmark that ships with Foveate is a library, not a leaderboard. Point it
at your documents and questions to see what Foveate does for *your* data.

## Pipelines

Each pipeline answers one question about one document with the same prompt,
JSON schema and citation check, so only the way evidence is chosen differs:

| Name | What it does |
|---|---|
| `full-context` | Sends the whole document. If it exceeds the window, no call is made and the result is an overflow failure. |
| `truncate` | Sends the first pages that fit the budget. |
| `naive-rag` | Fixed-size chunks, BM25 top-k up to the budget, one shot, no retry. |
| `foveate` | Retrieve, foveate, answer, verify, retry. |

Add your own by subclassing `foveate.bench.longdoc.Pipeline` and registering it
(`@Pipeline.registry.register("mine")`).

## Gold items

`GoldItem(id, kind, doc_name, question, expected, gold_pages, meta)`, kinds:

* `answerable`: expected answer and the pages that hold the evidence.
* `unanswerable`: the document lacks the answer; correct behaviour is to abstain.
* `paraphrase`: a reworded answerable question; used for consistency.
* `needle`: a synthetic fact inserted at a known depth (0-100%) for
  lost-in-the-middle sweeps.

Items are JSONL (`gold.read_items` / `gold.write_items`). Builders in
`foveate.bench.longdoc.gold` create unanswerable items, needle sweeps and
page-level evidence labels from your own documents.

## Metrics

Correctness is deterministic first (numeric match with scale words such as
"million"), then an LLM judge with a rubric. Also reported: citation page
precision and recall against gold pages, how many answers were grounded,
abstention accuracy on unanswerables, hallucination rate (answering an
unanswerable), run-to-run agreement across seeded runs and paraphrases, tokens,
estimated cost and latency.

## Start from your own files

You do not need FinanceBench. Give Foveate your documents and a handful of questions with
known answers; it builds the rest.

```python
from foveate import Document
from foveate.bench.longdoc import Config, Example, StaticCorpus, run, starter_items

documents = [Document.load("policy_2024.pdf"), Document.load("policy_2025.pdf")]
examples = [
    Example(
        doc_id="policy_2025",
        question="How many days of parental leave are granted?",
        expected="26 weeks",
        evidence="Employees are granted 26 weeks of paid parental leave.",
    ),
]
items = starter_items(documents, examples)   # answerable + unanswerable + needles
report = await run(items, StaticCorpus(documents), runtime, runtime, Config(runs=1))
print(report.to_markdown())
```

For each example you get an answerable item (with the page that holds the evidence, found by
text overlap), an unanswerable item that pairs the question with another document that does
not contain the answer, and needle items at five depths for every document. Thirty to a
hundred real questions are enough to see big differences; fewer than that and you are
reading noise.

## Run it

```python
from foveate.bench.longdoc import Config, Corpus, FinanceBench, run
# items: list[GoldItem]; corpus resolves each item's document
report = await run(items, corpus, answer_runtime, judge_runtime,
                   Config(pipelines=("full-context", "naive-rag", "foveate"),
                          budget=6000, runs=3))
print(report.to_markdown())
report.write("bench/results")
```

The FinanceBench loader downloads the public questions and filings into
`~/.cache/foveate` (override with `FOVEATE_CACHE_HOME`). The filings are not
redistributed; this repository ships only derived gold annotations.
FinanceBench is CC BY-NC 4.0.
