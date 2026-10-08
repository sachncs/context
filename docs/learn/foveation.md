---
title: "Foveation: how much of each page"
description: "The four fidelity tiers (full, condensed, outline, dropped), how the token budget is shared between them, and the options that change the result."
---

The word comes from the eye. The *fovea* is the small centre of your retina where vision is sharp;
everything around it is coarse. You do not read a whole scene in detail, you look sharply at the part
that matters and keep a rough sense of the rest. Foveate does the same with a document.

## The four tiers

Once pages are ranked ([selection](selection.md)), each page gets one of four treatments:

| Tier | What the model sees | Cost | Who gets it |
|---|---|---|---|
| **Full** | The whole page, verbatim | The page's full size | The best-ranked pages |
| **Condensed** | A short extract: the most useful sentences of the page | About 150 tokens | Neighbours of full pages, and the next-best pages |
| **Outline** | One line: page number and heading | A few tokens | Everything else that has text |
| **Dropped** | Nothing | Zero | Pages that do not fit even as an outline line |

The outline matters more than it looks. It tells the model the rest of the document exists and what
each part is about, so it can ask for more: if the answer is not in the full pages, the model can name
the outline pages it wants, and Foveate adds them for a second round.

## How the budget is shared

You give `Foveator` a **budget**: the most tokens the evidence block may use. If you do not, Foveate
takes the model's context window (from a built-in table, or `Runtime(context_window=...)`) minus a
reserve for the answer.

By default 60% of the budget goes to full pages, 25% to condensed ones, and the remainder to outline
lines. These are `FoveationConfig(fovea_share=0.6, parafovea_share=0.25)`. The allocation never
exceeds the budget; a page that is too big for the room left is skipped in favour of smaller ones, except the best-ranked page, which is cut to fit rather than dropped.

```python
from foveate import Foveator
from foveate.foveation import FoveationConfig

foveator = Foveator(
    runtime,
    budget=6000,
    foveation=FoveationConfig(neighbours=1, condensed_tokens=150),
)
```

## Two options that change what the model sees

**Order.** `FoveationConfig(order="edges")` puts the best-scoring full pages at the start and the end
of the evidence block and the weakest in the middle, because models use the middle of a long prompt
least reliably. The default, `"reading"`, keeps document order, which keeps the text coherent when
pages are consecutive.

**Query-aware condensing.** By default a condensed page keeps its most "typical" sentences. With
`FoveationConfig(query_aware=True)` it keeps the sentences that share rare words with the question, so
the sentence that might hold the answer survives. Both options are off by default because we have not
yet shown they improve answers on our benchmarks; try them on yours.

## Seeing the result before you pay

`Foveator.plan()` returns the allocation without calling the model:

```python
plan = foveator.plan("What were capital expenditures?", [report])
for page in plan.foveation.shown():
    print(page.page, page.tier.name, page.tokens_after)
print(plan.prompt_tokens, "of", plan.document_tokens, "tokens")
```

You can run this with `Runtime.without_llm()`; it raises if anything tries to call a model.

## When the budget is too small

A budget below about 1,500 tokens leaves no room for a full page plus the prompt. Foveate refuses
budgets under 256, and below roughly 1,500 you will see pages cut short. If the page with the answer
is reduced to an outline line at a small budget, the model will usually say "not found" rather than
guess: that is the safe failure, and the signal to raise the budget or improve selection.

## Next

[Grounded answers](grounding.md): what happens after the model reads the pages.
