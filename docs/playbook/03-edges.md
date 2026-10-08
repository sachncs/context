---
title: "3. Put the best evidence at the edges"
description: "Models use the start and end of a long prompt best. How to order evidence, and what we have and have not measured."
---

**Rule: when you must send a lot, put the most relevant material at the beginning and the end of the
prompt, and the least relevant in the middle.**

## The problem

Imagine reading a 40-page briefing and then answering a question. You remember the opening and the
last pages well; the middle blurs. Language models behave similarly. The paper that named the effect,
*Lost in the Middle* (Liu et al., 2023), found that models answer best when the relevant information is at
the start or end of the input and worse when it sits in the middle, even for models built for long
context.

Practitioners act on this. Uber's enhanced RAG assistant added a post-processing step that removes
duplicate chunks and "structures the context based on the positional order" before the model sees it
(2025).

## What to do

* **Rank, then arrange.** After selection, put the top-ranked evidence first and second-best last, then
  work inward, so the weakest lands in the middle.
* **Keep related text together.** If two pages are consecutive and read as one argument, splitting them
  to the two ends can hurt. The ordering rule matters most when you have many separate pieces.
* **Put the question near the end.** Instructions and the question belong where the model will weigh them,
  and putting the question last also keeps the earlier part of the prompt identical across questions,
  which helps caching ([rule 4](04-stable-prefixes.md)).
* **Do not rely on it.** Ordering is a nudge, not a fix. If the right text is not in the prompt, no
  ordering helps.

## In Foveate

```python
from foveate import Foveator
from foveate.foveation import FoveationConfig

foveator = Foveator(runtime, foveation=FoveationConfig(order="edges"))
```

`order="edges"` deals the full pages alternately to the front and the back by score. The default,
`"reading"`, keeps document order. The answer prompt always puts the question last.

## Honest status

We built the option but have **not** shown that it improves answers on our benchmarks: our retrieval
check measures which pages are chosen, and reordering does not change that. That is why it is off by
default. If your prompts carry many separate pieces of evidence, test it on your questions before
adopting it.

## Mistakes to avoid

* **Assuming newer models are immune.** Long-context benchmarks keep finding degradation with length
  (see *NoLiMa* in [Research](../research.md)).
* **Scrambling a story.** For narrative documents, reading order often beats ranked order.
