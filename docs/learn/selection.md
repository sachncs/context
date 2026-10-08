---
title: "Selecting pages"
description: "How Foveate ranks pages for a question: keyword search, embeddings, hybrid fusion, query expansion and re-ranking, and when to use each."
---

Selection answers one question: *out of all these pages, which are most likely to hold what this
question needs?* It is the step where most document-question systems quietly fail, because if the
right page never reaches the prompt, nothing later can recover it.

## From pages to chunks to page scores

Pages can be long, so Foveate searches over **chunks**: slices of a page of up to 256 tokens that
remember which page and heading they came from. A search returns the best chunks. Foveate then
folds chunk scores into **page scores**: a page scores its best chunk plus a tenth of every further
matching chunk, so a page with several relevant passages edges out a page with one.

## The retrievers

You choose with `Foveator(retrieval=...)`.

| Setting | How it works | Needs | Strong when | Weak when |
|---|---|---|---|---|
| `"bm25"` | Classic keyword ranking (Okapi BM25). Words that are rare in the document count more. | Nothing | The question uses the document's words: names, numbers, codes | The question and the text use different words |
| `"embedding"` | Turns text into vectors and ranks by similarity of meaning. | An embedding endpoint | Wording differs: "capital outlay" versus "purchases of property" | Exact identifiers and numbers |
| `"hybrid"` (default) | Runs both and fuses the rankings by reciprocal rank. Falls back to BM25 when no embedder is configured. | Optional embedder | Mixed questions; the safest default | Costs an embedding pass |

Add `rerank=True` to have the model re-score the top candidates; it helps when the first ranking is
close and costs one extra call.

## Configure an embedder

Any OpenAI-compatible `/embeddings` endpoint works. Some retrieval models need a hint about whether
text is a document or a query:

```python
import dataclasses

from foveate import Foveator, Runtime
from foveate.backends.embeddings import OpenAIEmbedder

embedder = OpenAIEmbedder(
    "nvidia/nemotron-3-embed-1b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="...",
    passage_options={"input_type": "passage"},
    query_options={"input_type": "query"},
)
runtime = dataclasses.replace(Runtime.from_env(), embedder=embedder)
foveator = Foveator(runtime, retrieval="hybrid")
```

If your embedding model lives on the same endpoint as your chat model, you can skip the code and set
`FOVEATE_EMBEDDING_MODEL` in the environment.

Vectors are cached on disk by model and text, so re-indexing an unchanged document is free.
`HashingEmbedder` is an offline stand-in for tests; it is not semantic.

## Query expansion

Questions and documents often use different words for the same thing. `Foveator(expand=2)` asks the
model for two alternative search phrasings, searches with each, and fuses the rankings. It costs one
short model call per question (cached for repeats). If the call fails, the original question is used
alone.

## What we measured

On the FinanceBench check in the [benchmarks](../benchmarks.md), counting how often the page holding
the evidence was sent in full:

| Retrieval | Evidence page sent in full |
|---|---|
| BM25 | 60% |
| BM25 + query expansion | 73% |
| Hybrid | 76% |
| Embeddings only | 87% |
| Embeddings + query expansion | 90% |

Two cautions from the same benchmark. First, better retrieval did **not** raise answer accuracy in our
second run: the model still has to read a financial statement correctly. Second, choosing whole pages
has a cost: a short sentence inside a long page scored lower than it did in a small chunk, and plain
chunk retrieval found hidden sentences that Foveate's page ranking missed. If your answers are short
facts inside long pages, test chunk-level selection against page-level on your data.

## Next

[Foveation](foveation.md): given the ranking, how much of each page goes in the prompt.
