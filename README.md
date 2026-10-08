# Foveate

**Context engineering for LLM apps: put the right pages in the window, and prove the answer.**

Give Foveate a 200-page report and a question. It sends the pages that matter in full, the pages
around them in short, and an outline of the rest. The answer comes back with the page and the exact
quote, and Foveate checks the quote against the page. If the document does not contain the answer, it
says so.

```python
from foveate import Document, Foveator, Runtime

report = Document.load("annual_report.pdf")                 # pip install "foveate[pdf]"
with Runtime.from_env() as runtime:
    answer = Foveator(runtime).ask(
        "What were capital expenditures in fiscal 2018?", [report]
    )

print(answer.text)        # "$1,577 million"
print(answer.pages)       # [("annual_report", 60)]
print(answer.grounded)    # True: the quote was found on page 60
```

Foveate has **no required dependencies beyond PyYAML**, talks to any OpenAI-compatible model server
(OpenAI, vLLM, Ollama, NVIDIA, Together) with the standard library, and plugs into Pydantic AI, Google
ADK, LangGraph and Strands Agents through small separate packages.

## Why

| Sending the whole document | With Foveate |
|---|---|
| May not fit the window, or is silently cut | Page-level selection inside a token budget you set |
| 70,000+ tokens on every question | A fraction of that, and a plan that shows it before you pay |
| Models read long prompts less reliably | The evidence arrives in full, the rest as an outline |
| No way to audit an answer | Page citations, each quote checked against the source |
| A confident answer when the file lacks the fact | An explicit "not found" |

## Does it work? A measured answer

30 real SEC-filing questions, the same prompt and citation check for every pipeline, answer model
`gpt-oss-20b`. The method, gold sets and every result, including the ones that do not favour Foveate, are
in [docs/benchmarks.md](docs/benchmarks.md).

| | Send everything | Truncate | Plain RAG | **Foveate** |
|---|---|---|---|---|
| Correct answers | 37% | 7% | 50% | **67%** |
| Filing too large for the window | 30% | 0% | 0% | 0% |
| Every cited quote found on its page | 74% | 67% | 76% | **85%** |
| Says "not found" when it should (20 questions) | 65% | 100% | 95% | **100%** |
| Mean prompt tokens | 71,227 | 10,334 | 10,637 | 16,378 |
| Same verdict across paraphrases and runs | 63% | 87% | 70% | 57% |

Thirty questions is small; read gaps under about 15 points as likely rather than certain. Foveate is no
better than plain retrieval at finding the evidence page, sends more tokens, and is less consistent across
reworded questions. A second run with embeddings and query expansion put the evidence page in the prompt far
more often but did not raise accuracy, and Foveate missed a hidden sentence in long pages that plain RAG
found. The roadmap lists these as the next things to fix.

## What you get

* **Documents**: PDF, DOCX, HTML, Markdown and text as numbered pages;
  `report.select("10-14,40")`, `report.around(40)`.
* **Selection**: keyword search, embeddings, hybrid fusion, query expansion, re-ranking.
* **Foveation**: full, condensed, outline and dropped tiers inside your budget.
* **Grounding**: page citations, checked quotes, retry with feedback, abstention.
* **Planning**: `Foveator.plan()` shows tokens and cost before any model call.
* **Compression** for chat history and tool output, mostly without a model.
* **Memory**: session notes, durable facts, recall by query.
* **Reliability**: cache, retries, timeouts, circuit breaker, rate limits, reasoning-model recovery.
* **Evolution**: learn a playbook of strategies from graded samples ([guide](docs/guides/evolving-playbooks.md)).
* **Safety**: fencing for untrusted text. **Evaluation**: a harness and a starter test set from your files.

## Install

```bash
pip install foveate                  # core
pip install "foveate[pdf]"           # read PDFs
pip install "foveate[tokenize]"      # exact token counts for OpenAI models
```

```bash
export FOVEATE_BASE_URL=https://api.openai.com/v1     # or your vLLM / Ollama URL
export FOVEATE_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
```

Agent frameworks, each a separate package: `foveate-pydantic-ai`, `foveate-adk`, `foveate-langgraph`,
`foveate-strands`. Python 3.10 to 3.13 on Linux, macOS and Windows.

No model yet? `Runtime.without_llm()` runs planning, page selection, memory and the model-free compression
methods offline.

## Learn more

* [Documentation](https://sachncs.github.io/foveate/): quickstart, concepts, a twelve-rule
  [playbook](docs/playbook/index.md) for newcomers, guides, reference.
* [Examples](examples/): runnable scripts, including one per agent framework.
* [Why context engineering](docs/why.md), [benchmarks](docs/benchmarks.md) and
  [research and sources](docs/research.md).
* [Architecture](ARCHITECTURE.md) and [contributing](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).

Apache-2.0.
