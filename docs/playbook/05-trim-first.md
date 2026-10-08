---
title: "5. Trim before you summarise"
description: "Dropping old turns versus summarising them: the trade-off, the safe order, and how to combine them."
---

**Rule: reach for the cheap, exact methods (keep the last N turns, drop what is stale) before the clever
ones (model-written summaries), and use summaries only when you must keep long-range memory.**

## The problem

A long conversation has to be shortened. There are two families of methods, and they fail differently.

| | Trimming (keep the last N turns) | Summarising (replace old turns with a summary) |
|---|---|---|
| Extra model call | None | At each refresh |
| Reproducible | Yes: same input, same output | No: depends on the summary |
| Long-range memory | Weak: a hard cut-off | Strong: a compact carry-forward |
| Failure | You lose old context | The summary can be wrong, or an error can propagate |
| Easy to debug | Yes | Only if you log the summary prompts and outputs |

That comparison is the OpenAI Agents SDK team's own: trimming is "deterministic and simple with no
summarizer variability"; summarisation risks "context distortion/poisoning" and means you must log the
summaries. Google's agent whitepaper lists the same ladder: keep the last N turns, token-based truncation,
then recursive summarisation, ideally computed in the background and persisted.

## What to do

1. **Start with a window.** Keep the newest turns verbatim. Most task-focused agents need little more.
2. **Never cut between a tool call and its result.** The model sees an orphan and misbehaves.
3. **Add summarisation for the old part only**, and keep recent turns verbatim.
4. **Give summaries structure.** Fixed headings (goal, decisions, facts, open items) are easier to check
   than free prose. The *Beyond Token Savings* study found that partial, structured rewriting that
   preserved recent history verbatim helped consistently across models.
5. **Have a fallback.** If the model that writes summaries is down, degrade to trimming instead of
   failing the agent.

## In Foveate

```python
context.compress("window", budget=6000)                  # exact and free
context.compress("ushape|extractive", budget=6000)       # summarise, fall back offline
context.compress("clear_tool_results+ushape", budget=6000)
```

`a+b` stops as soon as the budget is met; `a|b` falls back on failure. `HistoryCompressor` cuts at a turn
boundary so a tool call stays with its result.

## Mistakes to avoid

* **Summarising what you can still afford to keep.** Summaries lose detail; do not make them early.
* **Summarising summaries forever.** Errors compound; re-derive from the stored originals when you can
  ([offload](../learn/compression.md)).
* **Not logging summaries.** When an agent forgets something, you need to read what it was told.
