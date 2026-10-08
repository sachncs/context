---
title: "Foveate documentation"
description: "Foveate is a Python library for getting the right text in front of a language model: select, shape, compress, remember, verify and measure."
---

Foveate is a Python library for **context engineering**: deciding what a language model sees. Give it
a 200-page report and a question, and it sends the pages that matter in full, the pages around them
in short, and an outline of the rest. The answer comes back with the page and the exact quote, and
Foveate checks the quote against the page.

It has no required dependencies, talks to any OpenAI-compatible model server, and works with the
agent framework you already use.

## Where to start

| If you want to... | Go to |
|---|---|
| See it work in five minutes | [Quickstart](quickstart.md) |
| Understand the idea first | [What is context engineering?](learn/context-engineering.md) |
| Learn the rules of thumb | [The playbook](playbook/index.md) |
| Ask questions about long documents | [Long documents guide](guides/long-documents.md) |
| Build an agent | [Agents and tools](guides/agents-and-tools.md) |
| Run a small model locally | [Local models](guides/local-models.md) |
| Check it on your own data | [Evaluating your pipeline](guides/evaluating.md) |
| See the evidence | [Benchmarks](benchmarks.md) |

## What problem it solves

Sending a whole document to a model has four costs: it may not fit, it costs money every time, it is
slow, and the model reads long prompts less reliably. Plain retrieval fixes some of that but leaves
you with chunks that have no page numbers and answers nobody checks. [Why context
engineering](why.md) explains the trade-offs and where Foveate fits among the alternatives.

## What is in the box

| Area | What you get | Learn more |
|---|---|---|
| Documents | PDF, DOCX, HTML, Markdown and text as numbered pages | [Pages and documents](learn/pages-and-documents.md) |
| Selection | Keyword search, embeddings, hybrid, query expansion, re-ranking | [Selecting pages](learn/selection.md) |
| Foveation | Full, condensed, outline and dropped tiers inside a token budget | [Foveation](learn/foveation.md) |
| Grounding | Page citations, checked quotes, retry, abstention | [Grounded answers](learn/grounding.md) |
| Planning | Tokens and cost before any model call | [Planning and cost](learn/planning.md) |
| Compression | Chat history and tool output, with and without a model | [Compression](learn/compression.md) |
| Memory | Session notes, durable facts, recall | [Memory](learn/memory.md) |
| Reliability | Cache, retries, timeouts, circuit breaker, rate limits, reasoning-model recovery | [Reliability](learn/reliability.md) |
| Safety | Fencing and checks for untrusted text | [Safety](learn/safety.md) |
| Evaluation | Benchmarks and a starter test set from your own files | [Evaluating](guides/evaluating.md) |

## What it is not

* Not a vector database. It uses the retriever you configure and keeps vectors in a local cache.
* Not an agent framework. It plugs into [Pydantic AI, Google ADK, LangGraph and Strands
  Agents](guides/agents-and-tools.md) through small separate packages.
* Not a guarantee. The [benchmarks](benchmarks.md) show where it helps and where it does not.

## Status

Version 0.2.0, beta. The API may change between minor versions; the [changelog](../CHANGELOG.md) lists
what changed. Source and issues: [github.com/sachncs/foveate](https://github.com/sachncs/foveate).
