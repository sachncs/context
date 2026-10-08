---
title: "2. Send less, not more"
description: "Why a bigger prompt is usually a worse prompt, the four costs of long context, and how to choose a budget."
---

**Rule: give the model the smallest context that contains what it needs, not the largest that fits.**

## The problem

Context windows have grown from a few thousand tokens to hundreds of thousands, and it is tempting to
treat them as free storage: if it fits, send it. That mistake has four costs.

1. **It may not fit.** In our benchmark of real SEC filings, 30% were larger than a 131,000-token
   window.
2. **It costs money every time.** You pay for every token on every question.
3. **It is slower.** The model reads the whole prompt before it writes a word.
4. **It makes answers worse.** Anthropic's engineers put it this way: as the number of tokens in the
   context window increases, the model's ability to accurately recall information from that context
   decreases. They call it *context rot*. A 2026 study of long-running search agents found models
   "give up or provide uncertain incorrect answers long before exhausting the context window", and
   that this gets more common as context grows.

So more context is a trade, not a gift. Each extra token you send has to earn its place.

## What the evidence says here

In our benchmark (30 questions over real filings, `gpt-oss-20b`), sending the whole filing answered 37%
correctly, against 67% for Foveate sending a fraction of the pages. A hidden fact was found at 64,000
tokens of context with about 6% of the tokens. These are small samples and one model, so treat them as
an illustration of the direction, not a law. The direction matches the research.

## What to do

* **Pick a budget on purpose.** A 4,000 to 12,000-token evidence block is a good first range for
  question answering. Raise it only if the answer page is being left out.
* **Select before you send.** Rank the material for *this* question ([selection](../learn/selection.md)).
* **Keep the rest reachable.** Do not delete what you did not send: leave an outline or a tool so the
  model can ask for more ([rule 7](07-just-in-time.md)).
* **Re-check when models change.** A bigger window or a better model shifts the sweet spot.

## In Foveate

```python
foveator = Foveator(runtime, budget=6000)
plan = foveator.plan(question, [report])
print(plan.prompt_tokens, "of", plan.document_tokens)
```

`plan` shows what a question would send before you pay for it.

## Mistakes to avoid

* **Sending everything "to be safe".** It is the opposite of safe.
* **Setting the budget to the window.** The window is a ceiling.
* **Cutting blindly.** Truncating the end of a document is selection by position; it fails whenever the
  answer is later. In our run, truncation answered 7% correctly because the evidence was past the cut.
