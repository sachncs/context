---
title: "Benchmarks"
description: "How Foveate is measured against full context, truncation and plain RAG, and what the first full run found."
---

This page states how Foveate is measured and what the measurements found, including the
places where it does not win. Every number comes from a run in `bench/results/` that you
can repeat.

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

## Run 1: the default pipeline against three alternatives

* Date: 8 October 2026. Answer model and judge: `openai/gpt-oss-20b` with
  `reasoning_effort=low`, served by NVIDIA's OpenAI-compatible API. The judge is the same
  model as the answerer, which favours neither side but is a limitation.
* Sample: 30 FinanceBench questions, each from a different filing (stratified by question
  type), plus 60 paraphrases of them, 20 unanswerable questions and 10 needle items (2 filings at
  5 depths). Run 0 is deterministic; runs 1 and 2 are independent samples at temperature 0.7.
* Evidence budget for `truncate`, `naive-rag` and `foveate`: 12,000 tokens. `full-context`
  sends the whole filing; filings above the model window (131,072 tokens minus a margin)
  count as "could not run".
* Foveate settings: defaults (BM25 retrieval, page tiers, up to two rounds), no embeddings.
* Raw results: `bench/results/run1/` in the repository (`longdoc.json`, `summary.json`).

### Answer quality (30 questions, deterministic run)

| | full-context | truncate | naive-rag | foveate |
|---|---|---|---|---|
| correct | 37% | 7% | 50% | **67%** |
| correct when it could run | 52% | 7% | 50% | 67% |
| could not run (filing too large) | 30% | 0% | 0% | 0% |
| wrongly said "not found" | 10% | 90% | 17% | 13% |

The two sampled runs (temperature 0.7) gave the same order: Foveate 60% and 60%, plain
retrieval 43% and 47%, full context 43% and 40%, truncate 10% and 10%.

### Grounding (answers that were given)

| | full-context | truncate | naive-rag | foveate |
|---|---|---|---|---|
| every quote found on its cited page | 74% | 67% | 76% | **85%** |
| quotes found on the cited page | 73% | 75% | 84% | **91%** |
| cites a gold evidence page | 89% | 100% | 68% | 69% |

`truncate` and `full-context` look strong on "cites a gold page" because they only answer
when the evidence happens to be in what they were sent.

### Reliability

| | full-context | truncate | naive-rag | foveate |
|---|---|---|---|---|
| abstains on unanswerable questions (20) | 65% | 100% | 95% | **100%** |
| answers an unanswerable question anyway | 10% | 0% | 5% | 0% |
| same verdict across paraphrases and runs | 63% | 87% | 70% | 57% |
| needle found, all depths (10) | 100% | 20% | 100% | 100% |

### Cost and speed (calls that ran)

| | full-context | truncate | naive-rag | foveate |
|---|---|---|---|---|
| mean prompt tokens | 71,227 | 10,334 | 10,637 | 16,378 |
| median latency | 7.8 s | 3.8 s | 4.7 s | 7.0 s |

### What this shows

* Foveate answered more questions correctly than sending everything (67% against 37%), and
  than plain retrieval (50%), while sending about a quarter of the tokens that full context used on the filings it could fit.
  Sending everything fails on 30% of questions outright because the filing is larger than the
  window, and loses the facts it does send in the middle of long filings.
* It gave up cleanly: it never answered an unanswerable question, and 85% of its answers had
  every quote verified, against 74% to 76% for the alternatives.
* `truncate` shows why silently cutting a document is dangerous: 7% correct, 90% of the time it
  said "not found" because the evidence was past the cut.

### Where Foveate does not win

* **Finding the page.** Gold-page recall is 67% for Foveate and 66% for plain retrieval:
  no better. The pages that were missed are the ones whose wording differs from the question
  (for example "capital expenditure" against "purchases of property, plant and equipment").
  Keyword ranking cannot find those. Of Foveate's 34 wrong answers on the original questions
  (30 questions x 3 runs = 90 answers), 15 were "not found".
* **Consistency.** The same verdict across paraphrases and repeated runs is 57% for Foveate
  and 70% for plain retrieval. Reworded questions change which pages are retrieved, and
  Foveate's retry round adds a second source of variation.
* **Tokens.** Foveate sent 16,378 tokens on average against 10,637 for plain retrieval,
  because it adds condensed neighbours, an outline and sometimes a second round.

### What this does not show

* Thirty questions is small. Differences below about 15 percentage points are within noise,
  so the ranking of Foveate and plain retrieval on correctness should be read as likely, not
  certain. The 17-point gap to full context is larger but also uncertain.
* One answer model, one dataset (financial filings), English only, one budget.
* The model judge shares the answerer's blind spots. Numeric answers were checked by exact
  number match first; only non-numeric answers used the judge.
* 38 of the 150 FinanceBench questions could not be used (see Data above), so the set is
  biased towards filings that are still downloadable.

## Finding the evidence page: retrieval settings

Run 1 used BM25 only. This check plans the prompt for 90 questions (the 30 questions and
their 60 paraphrases) under each retrieval setting, with no answer model, and counts how often
the page holding the evidence is sent in full, or at least condensed. Evidence budget 12,000
tokens; embeddings are `nvidia/nemotron-3-embed-1b`; expansion asks the model for two
alternative search phrasings. Raw results: `bench/results/retrieval*/`.

| Retrieval | Evidence page sent in full | Sent in full or condensed |
|---|---|---|
| BM25 (run 1 setting) | 60% | 81% |
| BM25 + position-aware order | 60% | 81% |
| BM25 + query expansion | 73% | 89% |
| Hybrid (BM25 + embeddings) | 76% | 93% |
| Embeddings only | 87% | 97% |
| Hybrid + query expansion | 88% | 98% |
| Embeddings + query expansion | 90% | 98% |

The biggest single improvement is adding embeddings: the evidence page is in the prompt in
full 27 points more often than with BM25 alone on this data, where questions and filings use
different words for the same thing. Query expansion adds about 13 points to BM25 and a few
points on top of embeddings. Reordering pages changes what the model sees, not which pages
are chosen, so it does not move this measure. Whether better retrieval raises answer accuracy
and consistency is measured in the next run (it is not claimed here).

## Needle in a haystack

A fact is hidden at a known depth in a long text and the model is asked for it. Same pipelines,
answer model `gpt-oss-20b`, evidence budget 4,000 tokens, lengths 8k to 64k (multi: 16k to 64k),
depths 0%, 25%, 50%, 75%, 100%. Raw results: `bench/results/needle/`. The haystacks are
seeded synthetic prose, and for the NoLiMa-style case a book from the NoLiMa dataset
(Adobe Research License, noncommercial research use; downloaded at run time, not
redistributed).

| Test | Send everything | Truncate | Plain RAG | Foveate |
|---|---|---|---|---|
| One literal needle (40 cases) | 100% | 25% | 100% | 100% |
| Three needles, all required (12 cases) | 100% | 0% | 100% | 100% |
| NoLiMa-style non-literal needle (45 cases) | 0% | 2% | 2% | 2% |
| Mean prompt tokens, one literal needle | 30,674 | 2,907 | 841 | 2,654 |

* **Literal needles are solved.** For this model the classic test does not separate the
  pipelines: sending everything works up to 64k tokens. Foveate matches it while sending about
  6% of the tokens (3,612 against 64,862 at 64k). Truncation finds the needle only when
  it sits at the very start. Plain RAG is cheaper still on this test, because the question
  repeats the needle's words and a keyword match is enough.
* **Non-literal needles are not solved by anyone.** When the question shares almost no words
  with the needle (a character who lives next to a landmark; the question names the city),
  every pipeline scored 0 to 2%, including sending the whole text. This matches the NoLiMa
  paper's finding that such needles are hard. We also tried an inference-friendly prompt
  and embedding retrieval; neither changed the result for this model. This is a limit of the
  answer model at low reasoning effort, and in a debugging case the needle page was reduced to
  an outline line, so retrieval also needs work. We report it rather than drop it.

## Reproduce

```bash
python scripts/build_gold.py              # downloads filings, writes gold sets
python scripts/run_longdoc.py --dry-run   # sizes only, no model calls
python scripts/run_longdoc.py --n 30 --runs 3
python scripts/summarize_results.py bench/results/run1
python scripts/run_needle.py --family literal --lengths 8000,16000,32000,64000
python scripts/run_needle.py --family nonliteral --retrieval hybrid --inference \\
  --embedding-model nvidia/nemotron-3-embed-1b
python scripts/run_needle.py --family multi --needles 3
```
