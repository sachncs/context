---
title: "8. Separate working memory from long-term memory"
description: "Why scratch notes and durable facts should be stored, recalled and expired differently."
---

**Rule: keep what the agent is doing right now apart from what it should still know next month.**

## The problem

If everything the agent learns goes into one pile, two things go wrong. The pile grows until it crowds
the window, and old scratch work ("waiting for the approval email") is recalled long after it stopped
being true.

Google's agent whitepaper (November 2025) draws the line this way. A **session** holds the chronological
history and working state of one conversation. **Memory** is what is *extracted* from sessions and
*consolidated* so it persists across them. Extraction decides what is worth remembering; consolidation
merges new facts with old, resolving conflicts. Cloud agent platforms offer the same split as short-term
and long-term memory.

## What to do

* **Store scratch and facts separately**, with different lifetimes: scratch is deleted when the task
  ends; facts are kept.
* **Consolidate at the end of a task.** Turn notes into a few self-contained sentences; drop chatter.
* **Recall by relevance, with a budget.** Fetch the few notes that match the current question and fit
  them into a small slot; do not load the whole memory.
* **Label recalled memory as possibly out of date.** It is a hint, not ground truth.
* **Let facts be deleted.** Anything that can be remembered must be forgettable.

## In Foveate

```python
memory = Memory(FilesystemNotesStore("notes/"), runtime)
memory.remember("Deployments run in eu-west-1.")                 # fact
memory.remember("Checked the billing export", session="s1")     # scratch
memory.consolidate("s1")                                         # one model call, then notes are removed
memory.message("deployments", budget=200)                        # a system message for the prompt
```

See [Memory](../learn/memory.md) for the details.

## Mistakes to avoid

* **Storing secrets or personal data** in plain-text notes.
* **Trusting memory over the user.** If the user says something new, update the fact.
* **Recalling too much.** A memory slot that grows without limit is just another way to fill the window.
