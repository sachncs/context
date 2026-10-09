---
title: "FAQ"
description: "Plain answers to common questions about models, accuracy, grounding, PDFs, privacy, cost and the benchmarks."
---

## Basics

**Is this just RAG?**
Retrieval is one step of it. Foveate also keeps neighbouring pages in short form and the rest as an
outline, cites by page, checks each quote against the source, retries with feedback, and can say "not
found". Our benchmark compares it with plain RAG on the same prompt, and shows where plain RAG does as well
or better.

**Do I need a model to try it?**
No. Loading documents, selecting pages, planning cost, memory recall and the model-free compression
methods all run offline. Answering, `ppa` summaries and re-ranking call a model.

**Which models work?**
Anything behind an OpenAI-compatible API (OpenAI, vLLM, Ollama, NVIDIA, Together, gateways), or an
in-process vLLM engine. Larger models follow the citation format more reliably; test yours.

**Does it work with LangChain, Pydantic AI, Google ADK or Strands?**
Yes, through small separate adapter packages. Foveate itself depends on none of them.
See [Agents and tools](guides/agents-and-tools.md).

## Accuracy and trust

**Will it make my answers correct?**
Not by itself. It makes it likelier that the right text reaches the model and makes wrong answers easier to
catch. In our benchmark it answered more questions correctly than the alternatives we tried, and it also
missed things plain retrieval found. Measure on your data ([evaluating](guides/evaluating.md)).

**Why did I get `grounded=False`?**
The model gave no citation, or a quote that was not on the cited page (after ignoring case, spacing and
punctuation). The answer may still be right; the flag says it could not be verified. Some models do not
follow the citation format; try another, or a larger one.

**Does a verified citation prove the answer is right?**
No. It proves the quote is on that page. The quote may not support the claim. Show the quote to a person
when the answer matters.

**Why does it say "not found" when the answer is in the document?**
Usually because the page with the answer was not selected, or only reached the prompt as an outline line.
Look at `foveator.plan(question, docs)`, raise the budget, add embeddings, or try `expand=2`.

**Can a 1B model do this?**
We tried; the result was inconclusive. A 1B reasoning model ran out of room to think in a small window, and
a 30B-class model followed the citation format rarely. See the benchmarks page. Small models need a
non-reasoning setup, a window large enough for the prompt, and testing.

## Documents

**Can it read scanned PDFs?**
Not yet. Text extraction in `foveate-pdf` uses `pypdf`, which needs a text layer. Run OCR first. It is on the roadmap.

**What about tables?**
They are extracted as text, so row and column structure can be lost. Numeric questions work when the number
appears in the extracted text.

**How does it page a Word or text file?**
Into pseudo-pages of about 500 tokens at paragraph boundaries, or at form-feed characters if present.

## Cost, speed and privacy

**How much does it cost?**
Run `foveator.plan()` first: it reports prompt tokens and, if you configure prices, an estimated cost,
without calling anything. In our benchmark the prompts averaged 16,000 tokens against 71,000 for sending
the whole filing, but about 45% more than plain retrieval.

**Is my data sent anywhere?**
Only to the model and embedding endpoints you configure. Foveate sends no telemetry. The response cache is a
local file; put it on private storage or disable it.

**Does it need the internet?**
Only to reach your model. With a local server (vLLM or Ollama) nothing leaves your machine.

## The benchmarks

**Where does the benchmark data come from?**
FinanceBench (Patronus AI, CC BY-NC 4.0): public questions about real SEC filings, downloaded at run time from
the issuers' own sites. We build and commit only our own gold annotations. The NoLiMa needle sets are used
under the Adobe Research License (non-commercial research) and also downloaded at run time.

**Can I trust the numbers?**
They are real runs with small samples (30 questions, one answer model). The benchmarks page lists the limits
and the results that do not favour Foveate. Run the harness on your own documents.

**Why "Foveate"?**
The fovea is the small centre of the retina where vision is sharp. To foveate is to look sharply at what matters
and coarsely at the rest, which is what the library does with a document.
