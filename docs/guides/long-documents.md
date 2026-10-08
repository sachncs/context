---
title: "Long documents and page control"
description: "Load documents, pick pages yourself or let Foveate choose, plan costs and index once."
---

## Loading

```python
from foveate import Document

report = Document.load("10k.pdf")                        # real pages
manual = Document.load("manual.md", page_tokens=600)     # pseudo-pages
raw = Document.load(pdf_bytes, format="pdf", doc_id="q3_report")
```

Formats: `pdf` (pypdf), `docx` (python-docx), `html`, `markdown`, `text`. The
loader is chosen from the suffix or `format=`; add your own by subclassing
`foveate.documents.Loader` and registering it. Blank pages are kept so page
numbers match the printed document.

Document ids must be unique when you pass several documents; they appear in
every citation.

## Choosing pages yourself

```python
report.select("12-14,40")      # Document with only those pages
report.around(40, radius=2)    # 38..42
report.page(40)                # Page(number, text, tokens, headings)
report.token_count
```

A `Document` is immutable, so these return new documents. Use
`document.text(markers=True)` to get `[doc p.N]` markers in front of each page.

## Letting Foveate choose

```python
from foveate import Foveator

foveator = Foveator(runtime, budget=6000)
answer = foveator.ask("Which segment had the largest margin?", [report])
```

Useful options (all fields of `Foveator`):

| Option | Default | Effect |
|---|---|---|
| `budget` | model window minus a reserve | Prompt tokens per question |
| `retrieval` | `"hybrid"` | `bm25`, `embedding` or `hybrid` |
| `rerank` | `False` | Model re-scores the candidates |
| `k` | 24 | Chunks retrieved |
| `max_rounds` | 2 | Extra rounds when the answer is unsupported |
| `ungrounded` | `"flag"` | `"abstain"` replaces unverified answers by not-found |
| `foveation` | `FoveationConfig()` | Tier shapes (neighbour radius, condensed share) |

`budget` unset means "use the window of the model": the registry in
`foveate.models` knows common models, and `Runtime(context_window=...)` or
`FOVEATE_CONTEXT_WINDOW` always wins.

## Many questions, one document

Index once, ask many times:

```python
index = foveator.index([report])
for question in questions:
    print(foveator.ask(question, index).text)
```

## Plan before you pay

```python
plan = foveator.plan(question, [report])
plan.document_tokens, plan.prompt_tokens, plan.estimated_cost_usd
plan.fits_whole_documents          # would sending everything even fit?
```

Set prices with `Runtime(prices=...)` to get a cost estimate.

## Embeddings

```python
from foveate.backends.embeddings import OpenAIEmbedder
runtime = Runtime(..., embedder=OpenAIEmbedder("text-embedding-3-small"))
Foveator(runtime, retrieval="hybrid")
```

Without an embedder, `hybrid` runs BM25 only. A `HashingEmbedder` (no
network, lower quality) is available for tests.

## Several documents

Pass a list. Pages compete for the same budget and citations carry the
document id:

```python
foveator.ask("Compare capex in both filings", [filing_2018, filing_2019])
```

## When the answer is not there

Check `answer.abstained`. If you prefer never to show an unverified claim, set
`ungrounded="abstain"`.
