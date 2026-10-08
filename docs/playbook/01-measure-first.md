---
title: "1. Measure first"
description: "Why a small, honest test set beats intuition, how to build one in an afternoon, and what to measure."
---

**Rule: before you change how context is built, build a small test set and measure the starting point.**

## The problem

Most context decisions feel obviously right. "Send more pages, surely that helps." "A summary of the
chat is better than dropping old turns." Then you ship it and quality is the same, or worse, and you
cannot tell why because you never measured before.

Context engineering is full of effects that cut against intuition. In the *Beyond Token Savings* study
of agent compression (2026), policies that used a third of the tokens ran 20% to 80% *slower* because
of extra calls; policies with the same average success rate solved different sets of tasks, one gaining
7 tasks while losing 8; and the best policy for one model was the worst for another. You cannot reason
your way to those facts. You have to look.

## What to do

1. **Write 30 to 100 real questions** with known answers, taken from what your users actually ask.
   Fewer than that and you are reading noise: with 30 questions, differences under about 15
   percentage points are not reliable.
2. **Include questions the data cannot answer.** A system that always answers looks perfect on easy
   questions and fails the first time the document lacks the fact. Add a few of these ([rule
   10](10-abstain.md)).
3. **Record the evidence.** For each question, note which page holds the answer. That lets you measure
   whether *retrieval* worked separately from whether the *model* read it right. Both fail, in
   different ways, and the fix differs.
4. **Measure more than accuracy.** Track tokens, latency and cost per question, and whether the answer
   was verified. A method that wins on accuracy but sends ten times the tokens may not be a win.
5. **Change one thing at a time**, and re-run.

Uber's agent platform team made the same point from the other direction: teams ship agents and "defer
evaluation until later", so they built tooling that makes evaluations the default from day one
(2026).

## In Foveate

Build a starter set from your own files with no model:

```python
from foveate import Document
from foveate.bench.longdoc import Example, starter_items

items = starter_items(
    [Document.load("policy_2024.pdf"), Document.load("policy_2025.pdf")],
    [Example("policy_2025", "How many weeks of parental leave?", "26 weeks",
             evidence="Employees are granted 26 weeks of paid parental leave.")],
)
```

You get answerable questions with gold pages, unanswerable ones (your question paired with a document
that lacks the answer), and needle tests at five depths. Run them against several pipelines with
`foveate.bench.longdoc.run`. The [evaluating guide](../guides/evaluating.md) walks through it.

## Mistakes to avoid

* **Testing on the examples you tuned on.** Keep some questions aside.
* **Letting a model grade itself.** Use exact checks where you can (numbers, names), and a different
  model for the rest. Our own benchmark used the answer model as judge for non-numeric answers and
  says so as a limitation.
* **Trusting a single run.** Sampling makes answers vary; repeat and look at agreement.
* **Believing a leaderboard.** Public results use other data and other models. Measure on yours.
