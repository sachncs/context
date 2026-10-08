# Foveate

**Context engineering for LLM apps: put the right pages in the window, and prove the answer.**

Upload a 200-page filing and ask a question. Foveate shows the model the pages
that matter at full detail, the pages around them condensed, and an outline of
the rest, then checks that every citation in the answer really appears on the
page it names.

```python
from foveate import Document, Foveator, Runtime

document = Document.load("annual_report.pdf")              # pip install "foveate[pdf]"
with Runtime.from_env() as runtime:
    answer = Foveator(runtime).ask(
        "What were capital expenditures in fiscal 2018?", [document]
    )

print(answer.text)                  # "... $1,577 million"
print(answer.pages)                 # [("annual_report", 60)]
print(answer.grounded)              # True: every quote was found on its cited page
```

When the document does not contain the answer, you get an explicit
`abstained=True` instead of a confident guess.

## Why

| Without Foveate | With Foveate |
|---|---|
| Whole document exceeds the window, or silently truncates | Page-level selection under a token budget you set |
| 100k+ tokens per question, slow and costly | A fraction of the tokens ([measured](docs/benchmarks.md)) |
| Answers cannot be audited | `(document, page, quote)` citations, verified against the source |
| Wrong answers when the file lacks the fact | Abstention |
| Agents drown in tool output and history | Page tools, history compression, tool-output reducers |

<!-- RESULTS -->

## What you get

* **Documents**: PDF, DOCX, HTML, Markdown, text as page-addressable `Document`s;
  `doc.select("10-14,40")`, `doc.around(40, radius=2)`.
* **Selection**: BM25 (no dependencies), embeddings, hybrid fusion, model re-rank.
* **Foveation**: full / condensed / outline tiers within your token budget.
* **Grounding**: verified page citations, retry with feedback, abstention.
* **Plan**: `Foveator.plan(...)` shows tokens and cost before any model call.
* **Agents**: `read_pages`, `search_document`, `document_outline` tools and
  history compression for Pydantic AI, Google ADK and LangGraph.
* **Compression**: `Context(...).compress("ppa", budget=4000)` for chat history
  and `"tool_output"` for JSON/CSV/HTML/log results; several methods need no model.
* **Reliability**: retry with jitter, timeouts, circuit breaker, rate limits,
  caching, single-flight; typed events, metrics and OpenTelemetry spans.
* **Local models**: any OpenAI-compatible server, in-process vLLM, and a Docker
  recipe for vLLM on CPU.
* **Evaluation**: a long-document benchmark with gold sets you can rerun on your data.

## Install

```bash
pip install foveate                   # core, no required dependencies
pip install "foveate[pdf]"            # read PDFs
pip install "foveate[openai]"         # OpenAI SDK / vLLM / Ollama / NVIDIA endpoints
pip install "foveate[litellm]"        # 100+ providers through LiteLLM
pip install "foveate[frameworks]"     # Pydantic AI, Google ADK, LangGraph
```

Also: `docx`, `tokenize` (tiktoken), `vllm`. Python 3.10-3.13 on Linux, macOS and Windows.

Configure a model through the environment:

```bash
export FOVEATE_BACKEND=openai
export FOVEATE_BASE_URL=https://integrate.api.nvidia.com/v1
export FOVEATE_MODEL=openai/gpt-oss-20b
export FOVEATE_OPTIONS='{"reasoning_effort": "low"}'
export OPENAI_API_KEY=...
```

## Try it without a model

```python
from foveate import Document, Foveator, Runtime

document = Document.load(open("report.txt", "rb").read(), format="text", doc_id="report")
plan = Foveator(Runtime.without_llm(), budget=4000).plan("What was revenue?", [document])
print(plan.document_tokens, "->", plan.prompt_tokens, "tokens")
```

## Documentation

Start at [docs/index.md](docs/index.md): [quickstart](docs/quickstart.md),
[concepts](docs/concepts.md), guides for
[long documents](docs/guides/long-documents.md),
[agents](docs/guides/agents-and-tools.md),
[local models](docs/guides/local-models.md),
[evaluation](docs/guides/evaluating.md) and
[production](docs/guides/production.md), plus [benchmarks](docs/benchmarks.md),
[reference](docs/reference/index.md), [FAQ](docs/faq.md) and
[roadmap](docs/roadmap.md). Architecture notes are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). `make setup && make check` runs lint,
`mypy --strict` and the tests (90% coverage gate). Security issues:
[SECURITY.md](SECURITY.md).

Apache-2.0.
