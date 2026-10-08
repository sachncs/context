---
title: "FAQ"
description: "Short answers about models, grounding, PDFs, tables, privacy and the benchmark data."
---

**Is this just RAG?**
Retrieval is one step. Foveate also keeps neighbouring context condensed and
the rest as an outline, uses page-level citations, verifies quotes against the
source, retries with feedback, and can abstain. The benchmark compares it with
plain RAG on the same prompt.

**Does it need a model?**
Loading, selection, planning, history windowing, extractive compression,
tool-output reduction and offload need none. Answering, `ppa` and re-ranking
call a model.

**Which models work?**
Anything behind LiteLLM or an OpenAI-compatible API, or an in-process vLLM
engine. Larger models follow the citation format more reliably.

**Why did I get `grounded=False`?**
The model gave no citation, or a quote that is not on the cited page (after
whitespace/case/punctuation normalisation). The answer may still be right; the
flag says it could not be verified.

**Can it read scanned PDFs?**
No. Text extraction uses pypdf, which needs a text layer. OCR is on the
[roadmap](roadmap.md).

**Tables?**
They are extracted as text, so row/column structure may be lost. Numeric
questions still work when the number appears in the extracted text.

**Is my data sent anywhere?**
Only to the model endpoint you configure. The cache is local.

**Where does the benchmark data come from?**
FinanceBench (PatronusAI, CC BY-NC 4.0) public questions and the issuers' own
filings, downloaded at run time. We commit only our derived gold annotations.

**Why "foveate"?**
See the [index](index.md): sharp detail where it matters, coarse elsewhere.
