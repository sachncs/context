---
title: "Agents and tools"
description: "Page tools, history compression and tool-output reducers for Pydantic AI, ADK and LangGraph."
---

Agents fail on context for two reasons: they are handed whole documents, and
their history and tool output grow every turn. Foveate has a fix for each, for
Pydantic AI, Google ADK and LangGraph.

## 1. Give the agent page-level tools

`document_tools(documents)` returns three tools:

| Tool | Purpose |
|---|---|
| `document_outline(doc_id="")` | Page count, tokens and headings with page numbers |
| `search_document(query, k=5, doc_id="")` | Best pages with a snippet each (BM25) |
| `read_pages(pages, doc_id="")` | Full text of e.g. `"10-14,40"`, with `[doc p.N]` markers |

Replies are capped (`max_tokens`, default 4000) and say when they were cut.
Bad page requests return an `error:` string to the model instead of raising, so
the agent can correct itself.

```python
# Pydantic AI
from foveate.integrations import pydantic_ai as foveate_pai
agent = Agent(model, tools=foveate_pai.document_tools([document]))

# Google ADK
from foveate.integrations import adk as foveate_adk
agent = LlmAgent(..., tools=foveate_adk.document_tools([document]))

# LangGraph
from foveate.integrations import langgraph as foveate_lg
agent = create_react_agent(model, foveate_lg.document_tools([document]))
```

Without a framework, `foveate.integrations.tools.DocumentTools(documents)`
gives the same functions via `.functions()`.

## 2. Keep the history inside a budget

```python
compressor = foveate_pai.history_processor(runtime, budget=6000)
agent = Agent(model, capabilities=[ProcessHistory(compressor)])
```

```python
agent = LlmAgent(..., before_model_callback=foveate_adk.model_callback(runtime, 6000))
```

```python
node = foveate_lg.compression_node(runtime, budget=6000)
agent = create_react_agent(model, tools, pre_model_hook=node)
```

Only the older turns are compressed (default method `ushape|extractive`; use
`ppa` for best quality with a model). The latest `keep_last` messages stay
verbatim, and the cut never separates a tool call from its result. The stored
history is untouched: the model sees the compressed version.

## 3. Shrink tool output

Structured tool results can be reduced without a model:

```python
Context(messages, runtime).compress("tool_output", budget=3000)
```

JSON arrays and long strings are truncated with counts ("... 481 more items"),
CSV keeps the header and edges, HTML becomes text, repeated log lines collapse.
Combine it with other methods: `"tool_output+ushape"`.

## 4. Trace it

```python
from opentelemetry import trace
from foveate.tracing import TracingObserver
runtime = Runtime(..., observers=(TracingObserver(trace.get_tracer("foveate")),))
```

Backend calls, retries, cache lookups and compression steps become spans with
`gen_ai.*` token attributes.

The tool and history adapters are covered by no-model tests with the real
message classes of each framework (CI job `frameworks`).
