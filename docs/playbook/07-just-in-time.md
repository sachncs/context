---
title: "7. Pull context just in time"
description: "Give an agent a way to fetch what it needs when it needs it, instead of loading everything up front."
---

**Rule: do not load all the data into the prompt; give the agent lightweight handles and tools to fetch
the exact piece when it needs it.**

## The problem

The obvious design for a document assistant is to retrieve some text and put it in the prompt before the
model starts. That works for one-shot questions. It breaks down for agents: you do not know in advance
what the agent will need, so you either load too much (rule 2) or too little.

Anthropic's engineering guide describes the alternative: agents keep "lightweight identifiers" such as
file paths, stored queries or links, and use tools to load data at runtime, like a person who does not
memorise a library but knows where to look. A document has natural handles: its outline and its page
numbers.

## What to do

1. **Start small.** Put in the prompt only what is needed to begin: the task, an outline of what is
   available, the tools.
2. **Offer three kinds of tool**: one to *list* (an outline), one to *search* (find the right place), and
   one to *read* (fetch exactly that place).
3. **Cap every reply.** A tool that can return a whole book will. Limit its output and say when it was
   cut, so the agent asks for fewer pages.
4. **Return errors as text.** Let the agent correct a bad request instead of crashing.
5. **Make citations part of the output.** Page markers in tool replies let the agent cite what it read.

## In Foveate

`DocumentTools` gives an agent exactly this, as plain typed functions:

| Tool | What it does |
|---|---|
| `document_outline(doc_id="")` | Page count, tokens and headings with page numbers |
| `search_document(query, k=5, doc_id="")` | The best pages with a snippet each |
| `read_pages(pages, doc_id="")` | The text of `"10-14,40"`, each page marked `[doc p.N]` |

Foveate has no dependency on an agent framework, so the adapters live in small separate packages:
`foveate-pydantic-ai`, `foveate-adk`, `foveate-langgraph` and `foveate-strands`. Each is a few dozen
lines; see [Agents and tools](../guides/agents-and-tools.md).

## Mistakes to avoid

* **Giving a read tool no cap.** The agent will read everything and you are back to rule 2.
* **Search that only matches exact words.** Add embeddings if users and documents use different words.
* **No outline.** Without it the agent searches blind.
