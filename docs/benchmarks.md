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

## Run 2: embeddings and query expansion

The retrieval check above says better retrieval puts the evidence page in the prompt far more
often. Run 2 asks whether that turns into better answers. Same 30 questions, paraphrases,
unanswerable questions and needles as run 1, same model and 12,000-token budget, three runs.
Both pipelines now retrieve with embeddings (`nvidia/nemotron-3-embed-1b`); Foveate also writes
two alternative search phrasings per question. `full-context` and `truncate` do not retrieve
and were not repeated. Raw results: `bench/results/run2/`.

| | Plain RAG (embeddings) | Foveate (embeddings + expansion) |
|---|---|---|
| Correct answers | 47% | **60%** |
| Wrongly said "not found" | 30% | **3%** |
| Gold evidence page cited (recall) | 79% | 76% |
| Every cited quote found on its page | 86% | 86% |
| Abstains on unanswerable questions | 100% | 100% |
| Same verdict across paraphrases and runs | 53% | 50% |
| Needle found (10 items) | **100%** | 80% |
| Mean prompt tokens | **11,081** | 16,080 |
| Median latency | **6.3 s** | 10.5 s |

* **Retrieval improved, answers did not.** Gold-page recall rose from 66-67% to 76-79% for
  both pipelines, but correctness did not rise (Foveate 67% to 60%, plain RAG 50% to 47%, both
  within the noise of 30 questions). The evidence page reaching the prompt is necessary, not
  sufficient: the model still has to read a financial statement correctly.
* **Foveate's lead over plain RAG held** (60% against 47%), mostly because plain RAG said
  "not found" on 30% of answerable questions while Foveate's second round and condensed
  neighbours cut that to 3%.
* **Where Foveate lost.** It missed the hidden sentence in two of the four mid-document
  needle items (50% at 50% and 75% depth), where plain RAG found all of them, because it
  ranks whole pages and a short sentence inside a long page scores lower than in a small chunk.
  It also sent about 45% more tokens and took about 66% longer.
* **Consistency did not improve** (50% to 53% across paraphrases and runs).

## Compression: how far can the context shrink?

Each method gets the same token budget (the original size divided by the ratio) and a fact is
planted where naive compression loses it. Then the model answers from the compressed text.
Answer model `gpt-oss-20b`; **4 samples per cell**, so each percentage moves in steps of
25 points and small differences mean nothing. All methods here need no model call (the
model-based `ppa` strategy was too slow for this rate-limited key and is not in this sweep).
Raw results: `bench/results/compression/`. Tasks:

* **document**: one sentence hidden in the middle of about 6,000 tokens of prose.
* **history**: a setting stated once, early in a long chat.
* **atoms**: six facts of different importance stated once each in a long chat; we also
  count how many survive (critical atom recall), without asking the model.
* **tool**: one field in a large JSON tool result.
* **qa**: HotpotQA (public multi-hop questions, ten paragraphs each), scored by exact match
  and F1 as in the compression papers.

Correct answers at 4x and 16x compression:

| Task | Method | 4x | 16x |
|---|---|---|---|
| document | query-aware | 100% | 100% |
| document | selective | 100% | 100% |
| document | Foveate (4x only; needs 1,500 tokens) | 100% | n/a |
| document | extractive, truncate | 0% | 0% |
| history | extractive, query-aware | 100% | 100% |
| history | selective | 0% | 0% |
| history | truncate, window, drop-the-middle | 0-100%* | 0% |
| atoms (facts kept) | extractive, query-aware, selective | 100% | 100% |
| atoms (facts kept) | truncate, window, drop-the-middle | 0-100%* | 0% |
| tool | query-aware | 50% | 50% |
| tool | structure-preserving reducer (`tool_output`) | 0% | 0% |
| qa (F1 >= 0.5 or exact) | query-aware | 75% | 75% |
| qa | truncate | 75% | 50% |
| qa | extractive, selective | 25-50% | 0-25% |
| qa | uncompressed reference | 50% | n/a |

\* `truncate` keeps the beginning, so it passes while the planted fact is still within the
kept part and fails after; at 8x and 16x it could not reach the budget on a chat made of many
short messages (recorded as errors, counted as wrong).

What this suggests, with the caveat of four samples per cell and synthetic filler:

* **The compressor has to know the question.** Methods that look at the question or at word
  rarity (`query`, `selective`) kept the planted fact at 16x in the document task; methods that
  cut by position or by word frequency lost it. This matches the finding of query-conditioned
  compression papers.
* **No method wins everywhere.** `selective` kept the fact in a document but not in a chat;
  `extractive` kept it in a chat but not in a document. Picking a method per task, and
  measuring, matters; this is the "rankings change between tasks and models" result of the
  *Beyond Token Savings* study, seen on a small scale.
* **A structure-preserving reducer is the wrong tool when the answer is one specific row.**
  `tool_output` keeps a JSON's shape and cuts rows, so the row the question needs was usually
  gone. Use it to keep an agent's history small, and a query-aware method when you need a value.
* **On HotpotQA, query-aware compression kept answers usable down to 16x**: F1 0.77 against
  0.70 uncompressed on four questions, which is within noise but shows no loss.
* The model-based `ppa` strategy, the sweep over more samples, and a second answer model are
  not done yet.

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

## Small models and larger models

A claim worth testing: with the right context, a small model can match a much larger one. We
tried it on 12 FinanceBench questions plus 8 unanswerable ones, with embedding retrieval for
both retrieval-based pipelines, one run each, graded by `gpt-oss-20b`. This sample is far too
small for statistics; read it as a smoke test. Raw results: `bench/results/model-30b/` and
`bench/results/small-*/`.

**A 30B-class model** (`nvidia/nemotron-3.5-lightning-30b-a3b`, a mixture-of-experts model
with about 3B active parameters) behind each pipeline, 12,000-token budget:

| | Send everything | Plain RAG | Foveate |
|---|---|---|---|
| Correct answers | 58% (17% did not fit) | **75%** | 67% |
| Abstains on unanswerable questions | 50% | **100%** | 88% |
| Answers an unanswerable question | 12% | 0% | 12% |
| Every cited quote verified | 10% | 0% | **25%** |
| Mean prompt tokens | 78,501 | 13,044 | 24,929 |

With this model, plain retrieval did at least as well as Foveate on correctness and
abstention. The model rarely followed the citation format (0-25% verified), so grounding
could not do its job. Both retrieval pipelines beat sending the whole filing on correctness
and used far fewer tokens, which is the part of the claim these numbers do support.

**A 1B reasoning model** (`openbmb/MiniCPM5-1B`, served by vLLM on CPU with an 8,192-token
window, 4,000-token budget) produced almost no usable answers: in 83% to 92% of calls the
model spent its whole token allowance thinking and the window left no room to continue. That
run measures a limit of this setup (a reasoning model in a small window), not whether
context engineering helps a small model, so it supports no claim either way. It did lead to a
fix: Foveate now caps its "give the model more room to think" retries to what fits in the
window, and reports the failure instead of sending a request the server rejects.

A rerun with a non-reasoning small model (`Qwen2.5-1.5B-Instruct`, 16,384-token window) was
started but processed only three of the twenty items in the time available (CPU inference takes
about seven minutes per item), so it is not reported.

**So the claim that a 1B model can beat a 30B model is not established by our data.** What the
data do support is narrower: retrieval and page selection cut tokens by roughly 3x to 7x against sending
a whole filing and did not cost accuracy for the models we tried (20B and 30B-class); whether
the same holds for a model with 1B parameters needs a non-reasoning small model, a larger
window, and more than 12 questions.

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
