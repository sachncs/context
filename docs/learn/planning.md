---
title: "Planning and cost"
description: "See what a question would send, and what it would cost, before any model call; index once and ask many times."
---

Most systems tell you what a prompt cost after you have paid for it. `Foveator.plan()` tells you
before.

## Plan a question

```python
from foveate import Foveator, Runtime

foveator = Foveator(Runtime.without_llm(), budget=6000)
plan = foveator.plan("Which segment had the highest margin?", [report])

plan.document_tokens          # total size of the document(s)
plan.prompt_tokens            # estimated size of the prompt that would be sent
plan.fits_whole_documents     # would sending everything even fit the window?
plan.estimated_cost_usd       # 0.0 unless you configured prices
plan.foveation.shown()        # the full and condensed pages
```

A plan needs no model, no network and no key, so it is safe to run in a unit test or to show a user
before they confirm an expensive request. (If you enable `expand=` query expansion, planning makes
one small model call to write the alternative phrasings; with `Runtime.without_llm()` it skips them.)

## Put a price on it

Cost estimates need prices, because Foveate does not know your contract.

```python
from foveate.usage import Price, PriceTable

prices = PriceTable({"gpt-4o-mini": Price(0.15, 0.60, cached_per_million=0.075)})
runtime = Runtime.from_env()
runtime = dataclasses.replace(runtime, prices=prices)
```

`cached_per_million` is the rate for prompt tokens served from the provider's prompt cache. When a
provider reports cached tokens, `Answer.usage.cached_tokens` shows how many, and `cost_usd` uses the
cached rate for them. This matters: policies that cut tokens but break the prompt cache can cost more
than they save.

## Ask many questions of one document

Chunking and indexing happen once if you build an index:

```python
index = foveator.index([report])
for question in questions:
    print(foveator.ask(question, index).text)
```

With embeddings, the vectors for each chunk are computed on the first call and cached on disk.

## Choosing a budget

* Start from what the model can use well, not from what it can hold. A 4,000 to 12,000-token evidence
  block is a good starting range for question answering.
* Raise it if answers say "not found" while the plan shows the answer page as an outline line.
* Lower it if cost or latency matter more than recall.
* With no `budget`, Foveate uses the model's window minus a reserve for the answer. That is the
  largest budget, not the best one.

## Next

[Compressing conversations and tool output](compression.md).
