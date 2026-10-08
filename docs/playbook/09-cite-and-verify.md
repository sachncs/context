---
title: "9. Cite, then verify the citation"
description: "Why you should require a source for every answer and check it with a program, not with another model."
---

**Rule: make the model say where each answer came from, and check that the source says it.**

## The problem

Language models write fluent text whether or not they know the answer. A wrong answer in a confident
voice is the worst failure, because nobody checks it. If the model must point to the page and quote the
sentence, a person can check in seconds, and a program can check instantly.

Citations alone are not enough. Models also invent citations: a plausible page number with a quote that is
not on the page. So the second half of the rule matters as much as the first.

## What to do

1. **Ask for structure.** Require the answer plus, for each claim, the document, the page and a verbatim
   quote.
2. **Check the quote in code.** Does the text appear on that page (ignoring whitespace, case and
   punctuation)? This needs no model and cannot be talked into agreeing.
3. **Decide what a failure means.** Retry once with feedback ("your quote was not found on page 60"),
   then either flag the answer or refuse it, depending on how costly a wrong answer is.
4. **Track the share of verified answers.** It is a reliability number you can alert on.

Evidence that this kind of care pays off: Uber's on-call assistant, after moving from plain retrieval to
agents that rewrite queries, pick sources and refine context, reported a relative 27% increase in
acceptable answers and a relative 60% reduction in incorrect advice. In our benchmark, 85% of Foveate's
answers had every quote found on its cited page, against 74% to 76% for the alternatives.

## In Foveate

```python
answer = foveator.ask(question, [report])
answer.grounded      # True only if there is a citation and every citation is verified
answer.citations     # document, page, quote, verified
answer.support       # share of citations verified
```

Use `Foveator(ungrounded="abstain")` to turn unverified answers into "not found".

## What it does not prove

A verified quote proves the text is on the page, not that the text supports the answer. A model can
quote a true sentence that does not say what it claims. Quote checking removes fabricated evidence; it is
not a full fact-check. For high-stakes answers, show the quote to a human.

## Mistakes to avoid

* **Asking a second model "is this right?"** It shares the first model's blind spots and can be fooled
  by the same text.
* **Accepting any citation.** Verify the page number as well as the quote.
* **Hiding the citation from the user.** Show it; that is what makes the answer auditable.
