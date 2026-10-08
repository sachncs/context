---
title: "What is context engineering?"
description: "A plain-language introduction: what a context window is, why more text is not better, and the six decisions that make up context engineering."
---

If you have used a language model, you have already done context engineering, whether or not
you called it that. Every time you paste a document into a chat, trim a long conversation, or
decide which search results to include, you are deciding **what the model gets to see**. This
page explains why that decision matters more than most people expect, and gives you the
vocabulary used in the rest of the docs.

## The model only knows what is in front of it

A language model answers from two sources: what it learned during training, and what you put in
the **prompt** for this one request. It does not remember last week's conversation, it has never
seen your company's filings, and it cannot open your PDF. Anything it must know *now* has to be
inside the prompt.

The prompt has a size limit, called the **context window**. It is measured in **tokens**, which
are pieces of words: roughly four characters of English, so a page of text is about 500 to 700
tokens. A model with a 128,000-token window can read around 200 pages at once. That sounds like
the problem is solved: just send everything.

## Why "send everything" fails

There are four separate problems, and they compound.

**1. It may not fit.** A 10-K annual report often runs 80,000 to 400,000 tokens. In our benchmark, 30%
of the filings were larger than a 131,000-token window. A model that cannot take the whole file
either returns an error or, worse, your tooling silently cuts it off, and the answer you wanted
was in the part that was cut.

**2. It costs money on every question.** Providers charge per token read. A 70,000-token prompt
asked 1,000 times is 70 million tokens. Prompt caching helps, but only for the part of the prompt
that stays identical.

**3. It is slow.** The model has to read everything before it writes the first word. More input
means a longer wait for every answer.

**4. Accuracy drops as the prompt grows.** This is the least obvious problem. Researchers call it
*context rot*: as the number of tokens in the window increases, the model's ability to recall
information from that context decreases (Anthropic's engineering team states it that way, and
several 2026 papers measure it). Models also use the middle of a long prompt less reliably than
its beginning and end. A fact buried on page 83 of 200 is harder for the model to use than the
same fact on a short page by itself.

So the goal is not to give the model *as much* as possible. It is to give it *the right
amount of the right things*, in a form it can use and you can check.

## The six decisions

Context engineering is the set of decisions about what goes into the prompt. Foveate groups
them into six. Each has its own page in these docs.

| Decision | The question | In Foveate |
|---|---|---|
| **Select** | Out of everything available, which parts are relevant to *this* question? | [Selecting pages](selection.md) |
| **Shape** | How much detail does each selected part deserve? | [Foveation](foveation.md) |
| **Compress** | When it still does not fit, what can shrink without losing the facts that matter? | [Compression](compression.md) |
| **Remember** | What should persist across turns and sessions, and how does it come back? | [Memory](memory.md) |
| **Verify** | Is the answer actually supported by what the model was shown? | [Grounded answers](grounding.md) |
| **Measure** | Did that change help, on *your* data? | [Evaluating](../guides/evaluating.md) |

A system that only retrieves has made the first decision. A system that only trims chat history has
made the third. Real applications need all six, and the decisions interact: a better selection step
lets you use a smaller window; a verification step tells you when selection failed.

## An example

Suppose a user uploads a 200-page annual report and asks, "What were capital expenditures in
fiscal 2018?"

* **Send everything.** The prompt is 140,000 tokens. It may not fit; if it does, it is slow and
  expensive, and the model has to find one sentence among 200 pages.
* **Send the first few pages.** Cheap, but the answer is on page 60. The model says it does not
  know, or guesses.
* **Search, then send the best chunks.** Better. But the chunks have no page numbers, nothing checks
  the model's answer, and if the document does not contain the answer the model may invent one.
* **Context engineering.** Rank the pages for this question. Send the best ones in full, their
  neighbours in short, and the rest as a one-line outline. Ask the model to answer with the page
  and the exact quote. Check the quote against the page. If it does not match, retry once, then
  say "not found".

That last version is what `Foveator` does, and you can see its plan before spending a token:

```python
plan = foveator.plan("What were capital expenditures in fiscal 2018?", [report])
```

## Is it worth it?

Honest answer: it depends on the size of the problem. If everything you send is a few thousand
tokens and nobody needs to audit the answer, send it all; the machinery is not worth it.

When documents are long, questions are many, or answers must be checkable, it pays. In our
[benchmark](../benchmarks.md) of real SEC filings, Foveate answered 67% of questions correctly
against 37% for sending the whole filing (30% of filings did not fit), and abstained correctly on
every question the document could not answer. The same benchmark also shows where it does not
help: finding hidden facts in a few specific settings, and consistency across reworded questions.
The numbers and their limits are on the benchmarks page, and we recommend you run the evaluation
on your own documents before trusting anyone's results, including ours.

## Where to go next

* Try it in five minutes: [Quickstart](../quickstart.md).
* Learn the vocabulary one idea at a time: [Pages and documents](pages-and-documents.md).
* Read the practical rules, one page each: [The playbook](../playbook/index.md).
