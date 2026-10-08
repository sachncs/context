---
title: "Quickstart"
description: "Install Foveate, plan a 200-page question without a model, then get a cited answer."
---

Five minutes, from install to a cited answer.

## 1. Install

```bash
pip install "foveate[pdf,openai]"
```

Extras: `pdf` (pypdf), `docx` (python-docx), `openai` (OpenAI SDK, also for
vLLM/Ollama/NVIDIA endpoints), `litellm` (100+ providers), `tokenize`
(tiktoken), `vllm` (in-process engine), `frameworks` (Pydantic AI, ADK,
LangGraph).

## 2. See what a 200-page upload would cost (no model needed)

<!-- run -->
```python
from foveate import Document, Foveator, Runtime

pages = [f"Section {n}. " + "Routine operations text. " * 80 for n in range(1, 201)]
pages[39] += " Capital expenditures were 1,577 million dollars in fiscal 2018."
document = Document.load(
    "\f".join(pages).encode(), format="text", doc_id="annual_report"
)

foveator = Foveator(Runtime.without_llm(), budget=4000)
plan = foveator.plan("What were capital expenditures in fiscal 2018?", [document])

print(f"document: {plan.document_tokens:,} tokens")
print(f"would send: {plan.prompt_tokens:,} tokens")
for page in plan.foveation.shown():
    print(page.page, page.tier.name)
```

`Runtime.without_llm()` raises if anything tries to call a model, so this is a
safe dry run. The plan shows the full document size, what would be sent, and
which pages are shown at which fidelity.

## 3. Ask a question

Point Foveate at any OpenAI-compatible endpoint (or LiteLLM) with environment
variables, then ask:

```bash
export FOVEATE_BACKEND=openai
export FOVEATE_BASE_URL=https://integrate.api.nvidia.com/v1   # or your own server
export FOVEATE_MODEL=openai/gpt-oss-20b
export OPENAI_API_KEY=...
```

```python
from foveate import Document, Foveator, Runtime

document = Document.load("annual_report.pdf")          # needs foveate[pdf]
with Runtime.from_env() as runtime:
    answer = Foveator(runtime).ask(
        "What were capital expenditures in fiscal 2018?", [document]
    )

print(answer.text)
for citation in answer.citations:
    print(citation.doc_id, citation.page, citation.verified, citation.quote)
print("grounded:", answer.grounded, "tokens:", answer.usage.total_tokens)
```

`answer.grounded` is true only when every cited quote was found on the cited
page. When the document does not contain the answer, `answer.abstained` is true
and `answer.text` says so.

## 4. Look at specific pages yourself

```python
document.select("10-14,40")            # a Document with just those pages
document.around(40, radius=2)          # page 40 and its neighbours
document.page(40).text                 # one page
document.outline()                     # headings with page numbers
```

## 5. Use it inside an agent

```python
from foveate.integrations import pydantic_ai as foveate_pai

agent = Agent(model, tools=foveate_pai.document_tools([document]))
```

The agent can then call `document_outline`, `search_document` and
`read_pages` instead of receiving the whole file. See
[agents and tools](guides/agents-and-tools.md).

## Next

* [Concepts](concepts.md) - how foveation and grounding work.
* [Benchmarks](benchmarks.md) - what you gain, measured.
* [Evaluate your own pipeline](guides/evaluating.md).
