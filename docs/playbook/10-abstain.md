---
title: "10. Abstain when the evidence is missing"
description: "Why 'I could not find this' is a correct answer, how to test for it, and how to make it the default."
---

**Rule: when the documents do not contain the answer, the system should say so, and you should measure
that it does.**

## The problem

Ask a model a question the document cannot answer and it will often answer anyway, from its training or
by guessing. This is the most dangerous kind of hallucination in document work, because the answer looks
sourced. Suppose a user asks about a company's 2019 dividend and uploads the 2022 report: a helpful model
may state a plausible number.

Most test sets never catch this, because every question in them has an answer.

## What to do

* **Allow the model to say no.** The prompt must give it a clean way: a `found: false` field.
* **Test for it.** Pair real questions with documents that provably lack the answer and check that the
  system refuses. Aim for near 100%; a system that is right 90% of the time on answerable questions
  and invents answers on the rest is not safe to ship.
* **Treat refusal as a result.** Show "not found in these documents" to the user and, where useful, say
  which documents were searched.
* **Combine with verification** ([rule 9](09-cite-and-verify.md)): an answer with no verified quote
  should not be presented as sourced.

## What we measured

On 20 unanswerable questions in our benchmark, Foveate abstained correctly 100% of the time. Plain
retrieval got 95%, and sending the whole filing got 65%, answering 10% of them anyway. A larger
test is needed before relying on these exact numbers, but the pattern is the reason this rule exists.

## In Foveate

* The answer format has `found`; the model is told to set it to false when the pages lack the answer.
* `answer.abstained` is true when the system returned "I could not find this in the provided documents."
* `starter_items` builds unanswerable test items from your own documents automatically.

## Mistakes to avoid

* **Optimising only for answer rate.** A system that always answers looks best on a test with no
  unanswerable questions.
* **Letting the retriever hide the problem.** If retrieval returns the closest text no matter what, the
  model sees something to quote. Include a test for the case where the right page does not exist.
