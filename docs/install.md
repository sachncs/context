---
title: "Install and connect a model"
description: "Install Foveate, choose the optional extras, and point it at an OpenAI-compatible model server."
---

## Install

Foveate is not on PyPI yet; the `pip install foveate` form is coming soon. Until then, install from GitHub:

```bash
pip install "git+https://github.com/sachncs/foveate"
```

The core is pure Python with no dependencies and needs no model SDK. It handles text, Markdown and HTML,
and it does context selection, context management and compression. Everything that needs a third-party
library is a separate package, installed only if you want it.

| Add-on | Adds | Needed for |
|---|---|---|
| `foveate[tokenize]` | `tiktoken` | Exact token counts for OpenAI models (otherwise about four characters per token) |
| `foveate[vllm]` | `vllm` | Running a model in-process (Linux) |

An extra from GitHub looks like `pip install "foveate[tokenize] @ git+https://github.com/sachncs/foveate"`.

## File formats

PDF and Word support lives in plugin packages in `integrations/`. Install one and `Document.load` finds it
on its own (through the `foveate.loaders` entry point). There is nothing to import.

| Package | Adds | Reads |
|---|---|---|
| `foveate-pdf` | `pypdf` | `.pdf` |
| `foveate-docx` | `python-docx` | `.docx` |

```bash
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/pdf"
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/docx"
```

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

Each adapter lives in `integrations/<name>`. From GitHub, with the same `#subdirectory=` form as above:

```bash
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/pydantic-ai"   # Pydantic AI
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/adk"           # Google ADK
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/langgraph"     # LangGraph / LangChain
pip install "git+https://github.com/sachncs/foveate#subdirectory=integrations/strands"       # Strands Agents
```

The package names (`foveate-pydantic-ai` and so on) will install directly once they are on PyPI.

See [Agents and tools](guides/agents-and-tools.md).
