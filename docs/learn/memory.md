---
title: "Memory"
description: "Working notes for one task and durable facts across sessions: what to store, how recall works, and how to consolidate."
---

A model has no memory of its own. What looks like memory is context you put back in the prompt.
Deciding what to keep, and how to find it again, is memory engineering.

## Two layers

| Layer | What it holds | Lifetime |
|---|---|---|
| **Session notes** | Scratch work for the current task: "checked the billing export", "waiting for approval" | One session; discard them when it ends |
| **Facts** | Short statements worth keeping: "deployments run in eu-west-1", "the user prefers short answers" | Across sessions |

The split follows how agent platforms describe it: the session is the working memory of a
conversation; memory is what is extracted and consolidated so it survives across them.

## Using it

```python
from foveate import Memory
from foveate.stores import FilesystemNotesStore, MemoryNotesStore

memory = Memory(MemoryNotesStore())          # or FilesystemNotesStore("notes/") to persist
memory.remember("Deployments run in eu-west-1.")                  # a durable fact
memory.remember("Checked the billing export", session="s1")      # a session note

memory.recall("where do deployments run", k=3)       # best-matching notes, no model call
memory.message("deployments", budget=200)            # a system message that fits 200 tokens
```

`recall` ranks stored notes with BM25 (keyword search), so it is instant and free. `message` returns a
ready-to-use system message ("Relevant memory (may be outdated): ...") or `None` when nothing matches.
Facts are deduplicated by content: remembering the same sentence twice stores it once.

## Consolidate a session

At the end of a task, turn the scratch notes into facts with one model call:

```python
memory = Memory(store, runtime)
facts = memory.consolidate("s1", limit=8)    # returns the facts written; deletes the notes
```

The model is asked to keep only durable facts, decisions, preferences and open tasks, and to ignore
instructions inside the notes (they are data). If the reply is not the expected JSON, you get a
`ValidationError` and the notes are left untouched.

## Pitfalls

* **Stale facts.** Memory can be out of date; the message says so, and you should delete facts that
  change (`memory.forget(path)`).
* **Keyword recall misses synonyms.** "Where do we deploy?" will not match "region" unless a word is
  shared. Phrase facts the way users ask, or recall with a few variants.
* **Do not store secrets.** Notes are stored as plain text.

## Next

[Reliability](reliability.md): retries, caching and failure handling around every model call.
