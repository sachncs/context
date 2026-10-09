---
title: "Quickstart"
description: "Plan a question about a 200-page report without a model, then ask it with checked citations."
---

Five minutes. You need Python 3.10+ and Foveate (see [Install](install.md); PyPI is coming soon). Reading PDFs needs the `foveate-pdf` plugin.

## 1. See what a long document would cost, with no model

<!-- run -->
```python
from foveate import Document, Foveator, Runtime

pages = [f"Section {n}. " + "Routine operations text. " * 80 for n in range(1, 201)]
pages[39] += " Capital expenditures were 1,577 million dollars in fiscal 2018."
report = Document.load("\f".join(pages).encode(), format="text", doc_id="annual_report")

foveator = Foveator(Runtime.without_llm(), budget=4000)
plan = foveator.plan("What were capital expenditures in fiscal 2018?", [report])

print(f"document: {plan.document_tokens:,} tokens")
print(f"would send: {plan.prompt_tokens:,} tokens")
for page in plan.foveation.shown():
    print(page.page, page.tier.name)
```

`Runtime.without_llm()` raises if anything tries to call a model, so this is a safe dry run. You will
see the whole report is about 30,000 tokens, the plan sends about 4,000, and page 40 (where the fact is)
is sent in full with its neighbours in short form. [Foveation](learn/foveation.md) explains the tiers.

## 2. Connect a model

```bash
export FOVEATE_BASE_URL=https://api.openai.com/v1     # or your vLLM / Ollama URL
export FOVEATE_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
```

## 3. Ask with citations

```python
from foveate import Document, Foveator, Runtime

report = Document.load("annual_report.pdf")
with Runtime.from_env() as runtime:
    answer = Foveator(runtime, budget=6000).ask(
        "What were capital expenditures in fiscal 2018?", [report]
    )

print(answer.text)
for citation in answer.citations:
    print(citation.doc_id, citation.page, citation.verified, citation.quote)
print("grounded:", answer.grounded, "| abstained:", answer.abstained)
```

`answer.grounded` is true only when every quote was found on the page it names. If the document does
not contain the answer, `answer.abstained` is true and `answer.text` says so. See [grounded
answers](learn/grounding.md).

## 4. Look at pages yourself

```python
report.select("10-14,40")      # just those pages
report.around(40, radius=2)    # page 40 and its neighbours
report.outline()               # headings with page numbers
```

## 5. Try the other half: agents and long chats

```python
from foveate import Context, Message, Role

context = Context(messages, runtime)
smaller = context.compress("clear_tool_results+ushape", budget=6000)
```

and the page tools for an agent framework ([guide](guides/agents-and-tools.md)).

## Where next

* [What is context engineering?](learn/context-engineering.md) for the ideas behind this.
* [The playbook](playbook/index.md) for twelve practical rules.
* [Evaluate it on your own documents](guides/evaluating.md) before you trust any numbers, including ours.
