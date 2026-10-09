---
title: "12. Tell the model how much room it has"
description: "Why an agent plans better when it knows its context budget, what you can do today, and what Foveate does not yet do."
---

**Rule: let the model know how many tokens it has used and how many remain, so it can decide to
summarise, save notes or stop.**

## The problem

A human working on a long task watches the clock and the page count. A model usually has no idea it is
at 90% of its window. It keeps reading, then the request fails or the oldest content is cut without
warning. Giving the model the number turns a hidden limit into something it can plan around: it can
write down what matters before the window fills, choose a cheaper tool, or finish with what it has.

Anthropic describes this as context awareness: feedback on the remaining context capacity after each tool
call, alongside memory tools that let the model write notes before old context is cleared.

## What to do

* **State the budget.** A line in the system prompt: "You have about 30,000 tokens of context left."
* **Update it as you go.** Refresh the number each turn or tool call rather than stating it once.
* **Tell the model what to do near the limit.** Save notes (rule 8), summarise, or ask the user.
* **Do the same job in code regardless.** Do not rely on the model to protect itself; compress on a
  trigger ([rule 5](05-trim-first.md)).

## In Foveate

Foveate does not yet inject a budget note for you; this is on the [roadmap](../roadmap.md) as "context
awareness". You can do it today, because the numbers are available:

```python
plan = foveator.plan(question, [report])
note = f"Evidence budget: {plan.budget} tokens; this prompt uses about {plan.prompt_tokens}."
```

and for histories, `HistoryCompressor.count(history)` returns the current size, so you can add a
system line yourself before each model call.

## Mistakes to avoid

* **Stating the window instead of what remains.** "128k context" is not actionable.
* **Counting with the wrong tokenizer.** Use the model's tokenizer (the `foveate[tokenize]` extra, see [Install](../install.md));
  the default estimate is about four characters per token and can be off by 20%.
