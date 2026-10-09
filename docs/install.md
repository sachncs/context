---
title: "Install and connect a model"
description: "Install Foveate, choose the optional extras, and point it at an OpenAI-compatible model server."
---

## Install

```bash
pip install foveate
```

The core package is pure Python with no dependencies and needs no model SDK. Add extras for what you use:

| Extra | Adds | Needed for |
|---|---|---|
| `foveate[pdf]` | `pypdf` | Reading PDFs |
| `foveate[docx]` | `python-docx` | Reading Word files |
| `foveate[tokenize]` | `tiktoken` | Exact token counts for OpenAI models (otherwise about four characters per token) |
| `foveate[vllm]` | `vllm` | Running a model in-process (Linux) |

Python 3.10 to 3.13 on Linux, macOS and Windows.

## Connect a model

Foveate talks to any server that speaks the OpenAI chat-completions format: OpenAI, vLLM, Ollama,
NVIDIA's API, Together, Azure gateways and many others. Set three variables:

```bash
export FOVEATE_BASE_URL=https://api.openai.com/v1     # your server's /v1 root
export FOVEATE_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
```

Then build a runtime from them:

```python
from foveate import Runtime

with Runtime.from_env() as runtime:
    ...
```

| Provider | `FOVEATE_BASE_URL` | Notes |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | |
| Ollama | `http://localhost:11434/v1` | Any key value works |
| vLLM server | `http://localhost:8000/v1` | See [local models](guides/local-models.md) |
| NVIDIA | `https://integrate.api.nvidia.com/v1` | Many hosted models are reasoning models; set `FOVEATE_OPTIONS='{"reasoning_effort": "low"}'` |

Set `FOVEATE_CONTEXT_WINDOW` if your provider serves a smaller window than the model supports, so
budgets are computed correctly. The full list of settings is in the [reference](reference/settings.md).

## Check it without a model

You can try most of Foveate before configuring anything:

```python
from foveate import Runtime
runtime = Runtime.without_llm()      # raises if any model call is attempted
```

Planning, page selection, memory recall and the model-free compression methods all work with it.

## Agent frameworks

Foveate does not depend on any agent framework. Install the adapter for yours as a separate package:

```bash
pip install foveate-pydantic-ai     # Pydantic AI
pip install foveate-adk             # Google ADK
pip install foveate-langgraph       # LangGraph / LangChain
pip install foveate-strands         # Strands Agents
```

See [Agents and tools](guides/agents-and-tools.md).
