---
title: "6. Clear old tool output, keep the record"
description: "Why agent context fills with tool results, what is safe to remove, and why failures should stay."
---

**Rule: when old tool results are no longer needed, replace them with a short stub that says they
existed; never silently delete the fact that a call happened, and keep failures.**

## The problem

An agent calls tools: it searches, queries a database, reads a file. Each result goes into the
conversation, and results are big: JSON with hundreds of rows, a page of logs, a web page. After ten
calls most of the context is output the agent already used and will never need again. Meanwhile, the
agent still needs to remember *what it did*, so it does not repeat a search or forget a step.

Anthropic's engineers describe clearing old tool results as "one of the safest lightest touch forms of
compaction", and their platform replaces each cleared result with placeholder text so the model knows it
was removed. A 2026 study of expense-itemisation agents found that pruning to the last five tool calls
raised task completion from 71.0% to 79.0% while cutting tokens by about two thirds, and adding summaries
reached 91.6%.

## What to do

* **Clear oldest first**, and keep the newest few results intact.
* **Leave a stub**, not a hole: the tool name, how much was removed, and a few words of the content.
* **Keep errors.** A failed call records what does not work, and is usually short. The FOCUS paper (2026)
  adds a defensive pass for exactly this: spans carrying negative constraints or state are rescued even
  when they look unimportant. The Manus team makes the same point: leave the failed attempt in, so the
  model does not repeat it.
* **Exclude tools whose output is the task.** A "get_policy" result may need to stay.
* **Make results re-fetchable.** Clearing is safe when the agent can call the tool again.

## In Foveate

```python
context.compress("clear_tool_results", budget=6000)
# or, for finer control:
from foveate.compression import ClearToolResults
ClearToolResults(keep=3, keep_errors=True, exclude=("get_policy",))
```

It clears the oldest results until the context fits, keeps results that look like failures, never clears
excluded tools, and makes no model call. For large structured results you do want to keep, `tool_output`
shrinks JSON, CSV, HTML and logs in place while keeping their shape and saying what was cut.

## Mistakes to avoid

* **Using a shape-preserving reducer to find one value.** In our compression sweep, `tool_output` cut the
  row the question needed. Use a query-aware method to pull a specific value.
* **Clearing so aggressively that the agent loops** because it forgot it already tried something.
