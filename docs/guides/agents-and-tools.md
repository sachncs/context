---
title: "Agents and tools"
description: "Give an agent page tools, keep its history inside a budget, and shrink tool output, with Pydantic AI, Google ADK, LangGraph or Strands Agents."
---

Agents run into context problems that one-shot question answering does not: they are handed whole files,
their history grows with every turn, and each tool call adds another bulky result. Foveate addresses all
three, and it does so **without depending on any agent framework**.

## How the pieces fit

Foveate's core contains two framework-neutral parts in `foveate.agents`:

* `DocumentTools(documents)`: `read_pages`, `search_document` and `document_outline` as plain typed
  functions with docstrings.
* `HistoryCompressor` and `HistoryAdapter`: keep a message history inside a token budget; an adapter
  teaches the compressor one framework's message type.

Each framework has a small adapter package that wraps those pieces in the framework's own types:

| Framework | Package | Import |
|---|---|---|
| Pydantic AI | `foveate-pydantic-ai` | `foveate_pydantic_ai` |
| Google ADK | `foveate-adk` | `foveate_adk` |
| LangGraph / LangChain | `foveate-langgraph` | `foveate_langgraph` |
| Strands Agents | `foveate-strands` | `foveate_strands` |

They live in the `integrations/` directory of the repository. Each exposes the same two functions:
`document_tools([...])` and a history hook. Anything else (CrewAI, LlamaIndex, your own loop) can use
`DocumentTools(documents).functions()` directly, and a short `HistoryAdapter` for its message type.

## 1. Give the agent page tools

The agent gets three tools instead of the whole file:

| Tool | What it does |
|---|---|
| `document_outline(doc_id="")` | Page count, tokens and headings with page numbers |
| `search_document(query, k=5, doc_id="")` | The best pages with a snippet each |
| `read_pages(pages, doc_id="")` | The text of pages such as `"10-14,40"`, each marked `[doc p.N]` |

Replies are capped (`max_tokens`, default 4,000) and say when they were cut. A bad request returns an
`error:` string so the agent can correct itself rather than crash.

```python
# Pydantic AI
import foveate_pydantic_ai as foveate_pai
agent = Agent(model, tools=foveate_pai.document_tools([report]))

# Google ADK
import foveate_adk
agent = LlmAgent(..., tools=foveate_adk.document_tools([report]))

# LangGraph
import foveate_langgraph as foveate_lg
agent = create_react_agent(model, foveate_lg.document_tools([report]))

# Strands Agents
import foveate_strands
agent = Agent(model=model, tools=foveate_strands.document_tools([report]))
```

## 2. Keep the history inside a budget

Only the **older** turns are compressed. The latest `keep_last` messages stay verbatim, and the cut never
separates a tool call from its result. The default method summarises the old turns as one transcript and
falls back to an offline method if the model is unreachable, so compression never aborts the run.

```python
# Pydantic AI
agent = Agent(model, capabilities=[ProcessHistory(foveate_pai.history_processor(runtime, 6000))])
# Google ADK
agent = LlmAgent(..., before_model_callback=foveate_adk.model_callback(runtime, 6000))
# LangGraph
agent = create_react_agent(model, tools, pre_model_hook=foveate_lg.compression_node(runtime, budget=6000))
# Strands Agents
agent = Agent(model=model, conversation_manager=foveate_strands.conversation_manager(runtime, 6000))
```

Useful knobs, all optional: `method` (default `ushape|extractive`; use `"window"` for exact and free,
or `"clear_tool_results+ushape"`), `keep_last`, `trigger` (start compressing only above this fraction of
the budget) and `target` (compress down to this fraction, leaving headroom so the cache prefix stays
stable longer).

The stored history is not changed; the model sees the compressed version.

## 3. Shrink tool output

Structured results can be reduced without a model:

```python
Context(messages, runtime).compress("tool_output", budget=3000)
```

JSON arrays and long strings are truncated with counts ("... 481 more items"), CSV keeps the header and
the edges, HTML becomes text, repeated log lines collapse. Old results can be replaced by stubs with
`clear_tool_results`, which keeps errors. See [compression](../learn/compression.md) and
[rule 6](../playbook/06-clear-tool-output.md) for when to use which.

## 4. Watch it

```python
from opentelemetry import trace
from foveate.tracing import TracingObserver
runtime = dataclasses.replace(runtime, observers=(TracingObserver(trace.get_tracer("foveate")),))
```

Backend calls, retries, cache lookups and compression steps become spans with `gen_ai.*` token
attributes.

## Runnable examples

`examples/frameworks/` has one short, complete script for each framework. The adapters are tested in CI
against the real message classes of each framework, with no model call.

## Write an adapter for another framework

Subclass `HistoryAdapter`: `flatten(message)` returns plain `Message`s, `rebuild(messages)` returns the
framework's messages, and `starts_turn(message)` says where a user turn begins. The Strands adapter is
about sixty lines and is a good template (`integrations/strands/foveate_strands/__init__.py`).
