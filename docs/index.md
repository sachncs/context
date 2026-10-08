---
title: "Overview"
description: "What Foveate does, who it is for, and when to use it."
---

**Context engineering for LLM apps: put the right pages in the window, prove the answer.**

Foveate takes long documents (a 200-page 10-K, a manual, a contract), long
conversations and bulky tool output, and turns them into a prompt that fits the
model, costs less, and comes back with **page citations that are checked
against the source**.

The name comes from the eye: the *fovea* is the small centre of sharp vision.
To *foveate* is to keep full detail where it matters and coarse detail
everywhere else.

## Why use it

You uploaded 200 pages. Without Foveate one of three things happens:

| What happens | Consequence |
|---|---|
| The document does not fit the model window | A provider error, or silent truncation: the answer is missing from what the model saw |
| It fits, so you send all of it | 100k+ tokens per question: slow, expensive, and the model loses facts in the middle |
| You bolt on a retriever | Chunks with no page numbers, nothing checks the answer, no way to say "not in the document" |

With Foveate, each question gets the relevant pages at full fidelity, their
neighbours condensed, and a cheap outline of the rest. The model answers in
JSON with `(document, page, quote)` citations. Every quote is searched for on
the cited page. Unsupported answers are retried with feedback, then flagged or
turned into an explicit "not found".

See [benchmarks](benchmarks.md) for measured results, including where Foveate
does not help.

## Who it is for

* Developers building document Q&A, research, finance, legal or support tools.
* Agent builders whose context fills up with tool output and chat history.
* Teams that must be able to audit an answer (which page said that?).
* Anyone running small local models (4-32k windows) where "send everything"
  is impossible.

## When to use it (and when not)

Use it when documents are bigger than you want to pay to send, bigger than the
window, or when answers must be cited and verifiable.

Skip it when everything already fits comfortably in a few thousand tokens and
nobody needs to audit the answer: just send it.

## Where it runs

Python 3.10-3.13 on Linux, macOS and Windows. Works with any model reachable
through LiteLLM, any OpenAI-compatible server (vLLM, Ollama, NVIDIA, Together,
...), or an in-process vLLM engine. Plugs into Pydantic AI, Google ADK and
LangGraph. Compression that needs no model is built in.

## How to start

```bash
pip install foveate            # core, no required dependencies
pip install "foveate[pdf]"     # to read PDFs
```

Then follow the [quickstart](quickstart.md) (5 minutes), read the
[concepts](concepts.md), or jump to a guide:

* [Long documents and page control](guides/long-documents.md)
* [Agents and tools](guides/agents-and-tools.md)
* [Local models with vLLM](guides/local-models.md)
* [Evaluating your own pipeline](guides/evaluating.md)
* [Production](guides/production.md)

## What it contains

| Area | What you get |
|---|---|
| Documents | PDF, DOCX, HTML, Markdown, text loaders; page-addressable `Document` |
| Selection | BM25 (no dependencies), embeddings, hybrid fusion, LLM re-rank |
| Foveation | Full / condensed / outline / dropped tiers under a token budget |
| Grounding | Page citations verified against the source, retry, abstention |
| Planning | `Plan`: tokens and cost before any model call |
| Compression | Chat history and tool output: ppa, hierarchical, ushape, window, extractive, truncate, offload, tool-output reducers, pipelines |
| Agents | `read_pages` / `search_document` tools and history compression for Pydantic AI, ADK, LangGraph |
| Reliability | Retry, timeout, circuit breaker, rate limit, deadline, cache, single-flight |
| Observability | Typed events, metrics, OpenTelemetry spans |
| Safety | Prompt-injection fencing for retrieved text |
| Evaluation | A long-document benchmark with gold sets you can rerun |

Also: [reference](reference/index.md), [FAQ](faq.md), [roadmap](roadmap.md).
