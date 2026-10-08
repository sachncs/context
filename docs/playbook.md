---
title: "The twelve rules"
description: "A short rulebook for context engineering, with the source behind each rule and the Foveate setting that implements it."
---

These rules come from the [research and playbooks](research.md) Foveate is built on. Each
says what to do, why, how to do it in Foveate, and what goes wrong without it. Rules marked
*measured* link to a benchmark; the rest are practice reported by the sources.

## 1. Measure first, with a golden set

Build a few dozen real questions with known answers, including some the data cannot
answer, before tuning anything. Uber's agent platform made evaluations the default from day
one for this reason. **In Foveate:** [evaluate your own pipeline](guides/evaluating.md);
`foveate.bench.longdoc` builds answerable, unanswerable, paraphrase and needle items.
**Without it:** every change is a guess.

## 2. Send less, not more

Accuracy falls as the window fills ("context rot"), and cost and latency rise with it.
**In Foveate:** `Foveator(budget=...)` and `Foveator.plan()`. *Measured:*
[benchmarks](benchmarks.md). **Without it:** you pay for tokens that make answers worse.

## 3. Put the best evidence at the edges

Models use the start and end of a long prompt best. Uber's post-processor orders retrieved
context by position for the same reason. **In Foveate:** `FoveationConfig(order="edges")`.
**Without it:** the page that answers the question can sit in the middle where it is used least.

## 4. Keep prefixes stable and append-only

Prompt caches match prefixes. Put instructions and stable material first, the question
last, and serialise deterministically. **In Foveate:** the answer prompt ends with the
question; `Usage.cached_tokens` shows hits. **Without it:** a cache that never hits.

## 5. Trim deterministically before you summarise

Dropping old turns is cheap and reproducible; summaries carry the risk of distortion and
must be logged. **In Foveate:** `window` first, then `ushape` or `ppa`; compose with
`window+ushape`. **Without it:** summaries of summaries drift.

## 6. Clear re-fetchable tool output, keep the record

An old 20,000-token tool result rarely matters, but the fact that the call happened does.
**In Foveate:** `clear_tool_results`, `tool_output`. **Without it:** agents drown in their
own tool calls.

## 7. Pull context just in time

Give the agent identifiers and tools, not the whole corpus. **In Foveate:** `read_pages`,
`search_document`, `document_outline` for Pydantic AI, ADK and LangGraph.
**Without it:** every turn carries the whole file.

## 8. Separate working memory from long-term memory

Session notes are scratch; durable facts are consolidated and recalled by relevance.
**In Foveate:** `foveate.memory.Memory`. **Without it:** either nothing is remembered or
everything is.

## 9. Cite, and verify the citation

Ask for document, page and quote, then check that the quote is on the page.
**In Foveate:** `Answer.citations`, `Answer.grounded`, `Answer.support`.
**Without it:** you cannot tell a sourced answer from a plausible one.

## 10. Abstain when the evidence is missing

"Not found" is a correct answer when the pages do not contain it. **In Foveate:**
`Answer.abstained`, `ungrounded="abstain"`. *Measured:* abstention accuracy on
unanswerable questions in the [benchmarks](benchmarks.md).

## 11. Fence retrieved text as data

Documents can contain instructions. Mark them as data, stop them closing the fence, and
flag the obvious attempts. **In Foveate:** `foveate.fencing`. This is defence in depth, not
a guarantee; combine it with layered guardrails.

## 12. Tell the model how much room it has

Models plan better when they know the budget. Foveate does not do this yet; it is on the
[roadmap](roadmap.md) as *context awareness*.
