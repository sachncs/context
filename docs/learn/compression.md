---
title: "Compressing conversations and tool output"
description: "The compression methods Foveate ships, which need a model and which do not, how to combine them, and how to choose by measuring."
---

Documents are one source of long context. The other two are **chat history** (every turn the
user and assistant have exchanged) and **tool output** (what an agent's tools return: JSON, logs,
HTML, tables). Both grow without bound in a long-running agent. Compression keeps them inside a
token budget.

```python
from foveate import Context, Message, Role

context = Context(messages, runtime)
smaller = context.compress("ushape+extractive", budget=4000)
print(smaller.report)       # tokens before and after, steps, cost, whether anything was cut
```

`compress` never changes the original. It returns a new `Context` and guarantees the result fits the
budget: either it fits, or `BudgetExceededError` is raised, or (if you ask for
`Budget(4000, Overflow.TRUNCATE)`) the result is hard-truncated and the report says so.

## The methods

| Method | Needs a model | What it does | Good for |
|---|---|---|---|
| `window` | no | Drops the oldest turns; system messages stay | Chat where recent turns matter most |
| `truncate` | no | Cuts oversize messages (`keep="head"`, `"tail"` or `"middle"`) | A last-resort guarantee |
| `extractive` | no | Keeps the sentences made of the most frequent words, in order | Repetitive chat |
| `selective` | no | Drops the lowest-information phrases (rare words, numbers and names count more) | Prose with facts in it |
| `query` | no | Keeps the sentences that share rare words with a question you give (`query=...`, or the last user message) | When you know what will be asked |
| `tool_output` | no | Shrinks JSON, CSV, HTML and logs while keeping their shape, and says what was cut ("481 more items") | Agent tool results |
| `clear_tool_results` | no | Replaces the oldest tool results with a one-line stub; keeps errors; keeps the newest `keep` | Agent history |
| `offload` | no | Moves old turns to a notes store and leaves a pointer | Nothing may be lost |
| `ushape` | optional | Keeps the head and tail turns; summarises or drops the middle | Long chat |
| `ppa` | yes | Partition, summarise each part in parallel, combine (Wolf et al., 2026) | High-quality summaries |
| `hierarchical` | yes | Repeats `ppa` rounds until it fits | Very long input |

## Combine them

* `"a+b"` runs them in order and stops as soon as the budget is met. `"clear_tool_results+ushape"`
  first clears old tool output, then summarises only if that was not enough.
* `"a|b"` falls back to `b` if `a` fails, for example `"ppa|extractive"` so a model outage does not
  stop your agent.
* Options go in as keyword arguments, or per stage: `compress("ushape+ppa", budget=4000, ppa={"leaf_tokens": 512})`.

## Choose by measuring, not by reputation

No method wins everywhere. In our small sweep (four samples per cell; see the
[benchmarks](../benchmarks.md)):

* `query` and `selective` kept a fact hidden in the middle of a document at 16x compression; `truncate`
  and `extractive` lost it.
* `extractive` kept a setting stated once in a long chat at 16x; `selective` did not.
* The structure-preserving `tool_output` reducer lost the one row the question needed, because it
  keeps the shape of the JSON by cutting rows. Use it to keep an agent's history small, and a
  query-aware method when you need a specific value.

The same pattern appears in the research: *Beyond Token Savings* (2026) found that policies with
similar average success solved different tasks, and that rankings changed between models. So test
methods on your own task with the starter benchmark ([evaluating](../guides/evaluating.md)).

## Cost is more than tokens

A method that uses a third of the tokens can still be slower or dearer if it makes extra model calls
or breaks the provider's prompt cache. The compression report records the tokens the compressor itself
spent (`report.usage`). Two knobs on `HistoryCompressor` help: `trigger` (start compressing only above
this fraction of the budget) and `target` (compress down to this fraction, leaving headroom so the next
few turns do not trigger it again).

## Compress without a model

Everything marked "no" above runs offline. `Runtime.without_llm()` raises if any model call is
attempted, which makes it a good guard in tests:

```python
runtime = Runtime.without_llm()
context.compress("selective", budget=600)
```

## Next

[Memory](memory.md) for what to keep across turns and sessions.
