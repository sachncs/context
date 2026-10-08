# Evaluating your own pipeline

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
