---
title: "Roadmap"
description: "What ships in 0.1.0, what is being built, and what is being considered."
items:
  - title: "Page-level selection and foveation"
    status: shipping
    summary: "Load PDF, DOCX, HTML, Markdown or text, rank pages with BM25, embeddings or both, and send each page at the detail it deserves."
    why: "Models read less and answer better when the evidence is at full detail and the rest is an outline."
    source: "Lost in the Middle; Context Engineering 2.0 (arXiv:2510.26493)"
    url: "docs/concepts"
  - title: "Verified citations and abstention"
    status: shipping
    summary: "Every answer cites document, page and quote. Quotes are checked against the page; unsupported answers are retried, flagged or turned into not-found."
    why: "An answer you cannot audit is a guess."
    source: "Uber Genie: fewer incorrect answers with agentic RAG"
    url: "docs/concepts"
  - title: "Agent page tools"
    status: shipping
    summary: "read_pages, search_document and document_outline for Pydantic AI, Google ADK and LangGraph."
    why: "Let the agent pull pages when it needs them instead of receiving the whole file."
    source: "Anthropic: just-in-time retrieval"
    url: "docs/guides/agents-and-tools"
  - title: "History and tool-output compression"
    status: shipping
    summary: "Compress chat history with ppa, ushape, window and more. Shrink JSON, CSV, HTML and logs without a model."
    why: "Agents fill their windows with old turns and bulky tool results."
    source: "ACON (arXiv:2510.00615); Less Context, Better Agents (arXiv:2606.10209)"
    url: "docs/guides/agents-and-tools"
  - title: "Position-aware page order"
    status: building
    summary: "Put the strongest pages at the start and end of the evidence block and the weakest in the middle."
    why: "Models use the middle of a long prompt least."
    source: "Lost in the Middle; Context rot studies (arXiv:2606.29718)"
  - title: "Query-aware condensing"
    status: building
    summary: "Condense neighbouring pages by keeping the sentences most relevant to the question, not the most frequent words."
    why: "Better use of the same token budget."
    source: "Provence; AttentionRAG (arXiv:2503.10720); EnComp (arXiv:2603.09222)"
  - title: "Cache-friendly prompts"
    status: building
    summary: "Order prompts so the stable part comes first and report cached tokens, so provider prompt caches hit."
    why: "Cheaper and faster repeat calls."
    source: "Don't Break the Cache (arXiv:2601.06007); TokenPilot (arXiv:2606.17016)"
  - title: "Tool-result clearing"
    status: building
    summary: "Replace old, re-fetchable tool results with a stub that keeps the call and its arguments."
    why: "Cuts agent context growth without losing the record of what was done."
    source: "Anthropic context management; Manus engineering notes"
  - title: "Needle-in-a-haystack benchmark"
    status: building
    summary: "Length-by-depth grids with NoLiMa, a RULER-style generator and a LongBench v2 sample, for all four pipelines."
    why: "Shows where long context fails and whether Foveate helps."
    source: "NoLiMa (arXiv:2502.05167); BABILong (arXiv:2406.10149)"
  - title: "Compression benchmark"
    status: building
    summary: "Accuracy against compression ratio for each method on documents, needle cells, chat history and tool output."
    why: "The clearest test of whether less context can still answer correctly."
    source: "End-to-End Context Compression at Scale (arXiv:2606.09659)"
  - title: "Long-term memory"
    status: planned
    summary: "Working notes per session and consolidated facts across sessions, with recall by query."
    why: "Agents need memory that outlives one conversation."
    source: "Google: Context Engineering, Sessions and Memory; A-MEM (arXiv:2502.12110)"
  - title: "Context awareness"
    status: planned
    summary: "Tell the model how many tokens it has used and how many remain."
    why: "Models plan better when they know the room they have."
    source: "Anthropic context awareness"
  - title: "Evaluate your own documents in minutes"
    status: planned
    summary: "Build a starter gold set from your own files: answerable, unanswerable and needle questions."
    why: "Measure first; most teams skip it."
    source: "Uber: evaluation-first agent platform"
  - title: "Compression guideline tuning"
    status: planned
    summary: "Learn compression instructions from failures, using the evolution loop already in Foveate."
    why: "Compression that adapts to your task."
    source: "ACON (arXiv:2510.00615); ACE (arXiv:2510.04618)"
  - title: "Table-aware PDF extraction"
    status: planned
    summary: "Keep rows and columns when reading tables from PDFs."
    why: "Financial and scientific questions often live in tables."
    source: "FinanceBench error analysis"
  - title: "Streaming answers"
    status: planned
    summary: "Stream the answer while citations are verified at the end."
    why: "Faster perceived response."
  - title: "Vector database connectors"
    status: exploring
    summary: "Use Pinecone, pgvector, Qdrant or similar as the retriever."
    why: "Teams already have indexes."
  - title: "Scanned pages and images"
    status: exploring
    summary: "OCR and vision models for pages without a text layer."
    why: "Many real documents are scans."
  - title: "PII redaction"
    status: exploring
    summary: "Mask personal data before text leaves your machine."
    why: "Compliance."
  - title: "TypeScript SDK"
    status: exploring
    summary: "The same page-first workflow for JavaScript and TypeScript apps."
    why: "Reach beyond Python."
  - title: "Hosted evaluation dashboard"
    status: exploring
    summary: "A shared place to compare pipelines and track results over time."
    why: "Teams want history, not one-off runs."
---

Status of each item. The list is the same one shown on the site's Coming next page.

## Shipping in 0.1.0

* **Page-level selection and foveation.** Load PDF, DOCX, HTML, Markdown or text, rank pages with BM25, embeddings or both, and send each page at the detail it deserves.
* **Verified citations and abstention.** Every answer cites document, page and quote. Quotes are checked against the page; unsupported answers are retried, flagged or turned into not-found.
* **Agent page tools.** read_pages, search_document and document_outline for Pydantic AI, Google ADK and LangGraph.
* **History and tool-output compression.** Compress chat history with ppa, ushape, window and more. Shrink JSON, CSV, HTML and logs without a model.

## Being built now

* **Position-aware page order.** Put the strongest pages at the start and end of the evidence block and the weakest in the middle.
* **Query-aware condensing.** Condense neighbouring pages by keeping the sentences most relevant to the question, not the most frequent words.
* **Cache-friendly prompts.** Order prompts so the stable part comes first and report cached tokens, so provider prompt caches hit.
* **Tool-result clearing.** Replace old, re-fetchable tool results with a stub that keeps the call and its arguments.
* **Needle-in-a-haystack benchmark.** Length-by-depth grids with NoLiMa, a RULER-style generator and a LongBench v2 sample, for all four pipelines.
* **Compression benchmark.** Accuracy against compression ratio for each method on documents, needle cells, chat history and tool output.

## Planned

* **Long-term memory.** Working notes per session and consolidated facts across sessions, with recall by query.
* **Context awareness.** Tell the model how many tokens it has used and how many remain.
* **Evaluate your own documents in minutes.** Build a starter gold set from your own files: answerable, unanswerable and needle questions.
* **Compression guideline tuning.** Learn compression instructions from failures, using the evolution loop already in Foveate.
* **Table-aware PDF extraction.** Keep rows and columns when reading tables from PDFs.
* **Streaming answers.** Stream the answer while citations are verified at the end.

## Exploring

* **Vector database connectors.** Use Pinecone, pgvector, Qdrant or similar as the retriever.
* **Scanned pages and images.** OCR and vision models for pages without a text layer.
* **PII redaction.** Mask personal data before text leaves your machine.
* **TypeScript SDK.** The same page-first workflow for JavaScript and TypeScript apps.
* **Hosted evaluation dashboard.** A shared place to compare pipelines and track results over time.

Open an issue to vote for an item or to propose another.
