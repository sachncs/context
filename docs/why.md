---
title: "Why context engineering"
description: "What changes when the right context reaches the model, what we measured, and where Foveate sits among compression, retrieval and agent-memory approaches."
---

A model can only answer from what it is shown. Most teams spend their effort on the model and
send it whatever fits. Context engineering is the work of deciding what the model sees: which
pages, at what detail, in what order, from which memory, checked how. Done well it changes cost,
latency and answer quality at the same time, because all three depend on the same thing: how
many tokens the model has to read, and how likely the right ones are among them.

## What we measured

Everything below comes from the [benchmarks](benchmarks.md), with the answer model
`gpt-oss-20b`, and the page says where the sample is small.

| Lever | Measured effect |
|---|---|
| Send the right pages instead of everything | 67% of FinanceBench questions answered correctly against 37% for sending the whole filing (30% of filings did not fit the window at all) |
| Send fewer tokens | A hidden fact found at 64,000 tokens of context with about 6% of the tokens (3,612 against 64,862); the whole-filing prompt averaged 71,227 tokens against 16,378 |
| Rank pages with embeddings | The evidence page reached the prompt in full 87% of the time against 60% with keywords only |
| Rewrite the query | +13 points for keyword search by letting the model propose the document's own wording |
| Check the citation | 85% of answers had every quote found on its cited page, against 74% to 76% for the alternatives |
| Allow "not found" | 100% correct abstention on unanswerable questions; sending everything answered 10% of them anyway |

## What it does not fix

* Questions whose wording shares nothing with the evidence, tested with NoLiMa-style needles,
  were answered by no pipeline, including sending the whole text.
* Foveate was less consistent across reworded questions than plain retrieval in the first run
  (57% against 70%), and sent more tokens than plain retrieval. Better retrieval is the
  current work.
* A 30-question sample cannot separate small differences.

## Where Foveate sits

| Approach | Examples | Needs | Works with API models | Output you can audit |
|---|---|---|---|---|
| Learned latent compressors | IC-Former, SAC, ComprExIT, LCLM | Training and hidden states | No | No |
| KV-cache compression | SnapKV, KVzip | Access to the serving engine | No | No |
| Token or phrase pruning | Selective Context, LLMLingua | A small model, sometimes | Yes | Partly |
| Query-aware pruning for RAG | Provence, AttentionRAG | A trained pruner | Yes | Partly |
| Agent context managers | ACON, FOCUS, vendor context editing | An agent harness | Yes | Partly |
| Memory services | Bedrock AgentCore, ADK memory | A hosted service | Yes | Varies |
| **Foveate** | Selection, tiers, compression, memory, verification, evaluation in one library | Python | **Yes** | **Yes: pages and quotes** |

Foveate does not replace the others. It is the layer that decides what text goes to any model,
shows its work (the plan, the pages, the quotes), and ships the evaluation harness to check it
on your data. If you host the model you can add a learned compressor behind it; if you use an
API, there is nothing to train.

## Start

1. Run [`Foveator.plan()`](quickstart.md) on your own file: no model, no cost.
2. [Evaluate](guides/evaluating.md) on 30 to 100 of your questions before changing anything.
3. Turn on embeddings and, if wording differs, query expansion; keep what the numbers support.
