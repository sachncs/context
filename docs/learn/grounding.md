---
title: "Grounded answers"
description: "How Foveate makes the model cite page and quote, checks every quote against the source, retries once, and abstains when the evidence is missing."
---

A fluent answer is not a true answer. Language models produce plausible text whether or not the
evidence supports it. **Grounding** is the practice of tying each claim to a place in the source, so a
person (or a program) can check it.

## What the model is asked to return

The model must reply with JSON:

```json
{
  "found": true,
  "answer": "$1,577 million",
  "citations": [
    {"doc": "3M_2018_10K", "page": 60, "quote": "Purchases of property, plant and equipment (1,577)"}
  ],
  "need_pages": []
}
```

* `found` is false when the shown pages do not contain the answer.
* Each citation gives the document, the page, and words **copied from that page**.
* `need_pages` lists outline pages the model wants to read before answering.

## How the quote is checked

Foveate does not ask a second model whether the answer looks right. It checks the quote itself,
deterministically: it looks for the quote on the cited page, ignoring case, spacing and punctuation
differences, and accepts a long near-match (a longest-common-subsequence ratio of at least 0.85) to
tolerate small extraction differences. A citation is `verified` only if that check passes against the
real page text.

The result on `Answer`:

| Field | Meaning |
|---|---|
| `text` | The answer, or the not-found message |
| `citations` | Each with `doc_id`, `page`, `quote`, and `verified` |
| `grounded` | At least one citation, and every citation verified |
| `support` | The share of citations that were verified (0 to 1) |
| `abstained` | The answer is the not-found message |
| `needs_more_context` | The model asked for pages that were never shown, and the answer is not grounded |
| `usage`, `cost_usd`, `seconds`, `rounds` | What it took |

Because the check is a program, a document cannot talk it out of its verdict: an instruction hidden in
a page cannot make a false quote pass.

## What happens when verification fails

1. **Round two.** If the answer is missing a citation, or a quote did not verify, or the model asked for
   outline pages, Foveate retries once. The prompt now includes a short note saying what was wrong, the
   requested pages are added, and the cited pages are kept in full.
2. **After the last round**, you choose the policy with `ungrounded=`:
   * `"flag"` (default): return the answer with `grounded=False`. The answer may still be right; you
     are told it could not be verified.
   * `"abstain"`: replace it with "I could not find this in the provided documents."

`max_rounds=1` disables the retry.

## Abstaining is a feature

If the document does not contain the answer, "not found" is the correct output, and a confident guess
is a hallucination. In our [benchmark](../benchmarks.md), Foveate abstained correctly on 100% of 20
unanswerable questions, while sending the whole filing to the model answered 10% of them anyway.

## What grounding does not do

* It checks that the quote **exists on the cited page**, not that the quote **proves the answer**. A
  model can quote a real sentence that does not support its claim. Quote presence removes invented
  citations; judging whether the evidence is sufficient is a harder problem and is on the
  [roadmap](../roadmap.md).
* It depends on the model following the output format. A 30B-class model we tested produced verified
  citations on only 0 to 25% of answers, so grounding could not do its job. Test your model first.
* It cannot rescue retrieval. If the right page was never shown, the answer will be "not found".

## Next

[Planning and cost](planning.md).
