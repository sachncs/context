---
title: "The playbook"
description: "Twelve practical rules of context engineering, each with the problem, the reason, what to do, and how to do it in Foveate."
---

This is a rulebook for anyone who sends text to a language model and wants it to answer well, cheaply
and checkably. It is written for people new to the subject: each rule has its own page that explains
the problem in plain words, why it happens, what to do, a worked example, and the mistakes to avoid.

The rules come from published research and from the engineering write-ups of teams that run agents in
production (see [Research and sources](../research.md)). Where a rule is backed by a measurement from
this project, the page says so, and where it is only reported practice, the page says that too.

If you are new to the idea, read [What is context engineering?](../learn/context-engineering.md)
first. It takes ten minutes.

## The twelve rules

| # | Rule | One line |
|---|---|---|
| 1 | [Measure first](01-measure-first.md) | Build a small test set before you change anything |
| 2 | [Send less, not more](02-send-less.md) | Longer prompts cost more and answer worse |
| 3 | [Put the best evidence at the edges](03-edges.md) | Models read the middle least reliably |
| 4 | [Keep prefixes stable](04-stable-prefixes.md) | Prompt caches match from the start of the prompt |
| 5 | [Trim before you summarise](05-trim-first.md) | Cheap and exact beats clever and drifting |
| 6 | [Clear old tool output, keep the record](06-clear-tool-output.md) | Agents drown in their own results |
| 7 | [Pull context just in time](07-just-in-time.md) | Give identifiers and tools, not the whole corpus |
| 8 | [Separate working memory from long-term memory](08-memory.md) | Scratch notes are not facts |
| 9 | [Cite, then verify the citation](09-cite-and-verify.md) | A source you did not check is not a source |
| 10 | [Abstain when the evidence is missing](10-abstain.md) | "Not found" is a correct answer |
| 11 | [Fence untrusted text](11-fence-untrusted-text.md) | Documents can contain instructions |
| 12 | [Tell the model how much room it has](12-room.md) | Planning needs a budget |

## How to use it

* **Building something new?** Read rules 1, 2, 9 and 10 first. They decide whether your system is
  trustworthy at all.
* **Fixing cost or latency?** Rules 2, 4, 5 and 6.
* **Building an agent?** Rules 6, 7, 8 and 11.
* **Not sure which applies?** Rule 1 will tell you.

Every rule links to the Foveate feature that implements it. None of the rules requires Foveate; they
describe good practice whatever tools you use.
