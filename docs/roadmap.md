---
title: Roadmap
description: What ships in 0.2.0, what the benchmarks say to fix next, and what is being considered.
items:
- title: Page-level selection and foveation
  status: shipping
  summary: Load PDF, DOCX, HTML, Markdown or text, rank pages with BM25, embeddings or both, and send each page at the detail it deserves.
  why: Models read less and answer better when the evidence is at full detail and the rest is an outline.
  source: Lost in the Middle; Context Engineering 2.0 (arXiv:2510.26493)
  url: docs/concepts
- title: Verified citations and abstention
  status: shipping
  summary: Every answer cites document, page and quote. Quotes are checked against the page; unsupported answers are retried, flagged or turned into not-found.
  why: An answer you cannot audit is a guess.
  source: 'Uber Genie: fewer incorrect answers with agentic RAG'
  url: docs/concepts
- title: Agent page tools
  status: shipping
  summary: read_pages, search_document and document_outline for Pydantic AI, Google ADK and LangGraph.
  why: Let the agent pull pages when it needs them instead of receiving the whole file.
  source: 'Anthropic: just-in-time retrieval'
  url: docs/guides/agents-and-tools
- title: History and tool-output compression
  status: shipping
  summary: Compress chat history with ppa, ushape, window and more. Shrink JSON, CSV, HTML and logs without a model.
  why: Agents fill their windows with old turns and bulky tool results.
  source: ACON (arXiv:2510.00615); Less Context, Better Agents (arXiv:2606.10209)
  url: docs/guides/agents-and-tools
- title: Position-aware page order
  status: shipping
  summary: Put the strongest pages at the start and end of the evidence block and the weakest in the middle.
  why: Models use the middle of a long prompt least.
  source: Lost in the Middle; Context rot studies (arXiv:2606.29718)
- title: Query-aware condensing
  status: shipping
  summary: Condense neighbouring pages by keeping the sentences most relevant to the question, not the most frequent words.
  why: Better use of the same token budget.
  source: Provence; AttentionRAG (arXiv:2503.10720); EnComp (arXiv:2603.09222)
- title: Cache-friendly prompts
  status: shipping
  summary: Order prompts so the stable part comes first and report cached tokens, so provider prompt caches hit.
  why: Cheaper and faster repeat calls.
  source: Don't Break the Cache (arXiv:2601.06007); TokenPilot (arXiv:2606.17016)
- title: Tool-result clearing
  status: shipping
  summary: Replace old, re-fetchable tool results with a stub that keeps the call and its arguments.
  why: Cuts agent context growth without losing the record of what was done.
  source: Anthropic context management; Manus engineering notes
- title: Needle-in-a-haystack benchmark
  status: shipping
  summary: Length-by-depth grids with NoLiMa, a RULER-style generator and a LongBench v2 sample, for all four pipelines.
  why: Shows where long context fails and whether Foveate helps.
  source: NoLiMa (arXiv:2502.05167); BABILong (arXiv:2406.10149)
- title: Compression benchmark
  status: shipping
  summary: Accuracy against compression ratio for each model-free method on documents, chat history, tool output and HotpotQA, with fact-retention metrics. Model-based methods are next.
  why: The clearest test of whether less context can still answer correctly.
  source: End-to-End Context Compression at Scale (arXiv:2606.09659)
- title: Long-term memory
  status: shipping
  summary: Working notes per session and consolidated facts across sessions, with recall by query.
  why: Agents need memory that outlives one conversation.
  source: 'Google: Context Engineering, Sessions and Memory; A-MEM (arXiv:2502.12110)'
- title: Context awareness
  status: planned
  summary: Tell the model how many tokens it has used and how many remain.
  why: Models plan better when they know the room they have.
  source: Anthropic context awareness
- title: Evaluate your own documents in minutes
  status: shipping
  summary: 'Build a starter gold set from your own files: answerable, unanswerable and needle questions.'
  why: Measure first; most teams skip it.
  source: 'Uber: evaluation-first agent platform'
- title: Compression guideline tuning
  status: planned
  summary: 'Learn compression instructions from failures: compare runs where full context succeeded and compressed context failed, and rewrite the guideline.'
  why: Compression that adapts to your task instead of one fixed prompt.
  source: ACON (arXiv:2510.00615); ACE (arXiv:2510.04618)
- title: Table-aware PDF extraction
  status: planned
  summary: Keep rows and columns when reading tables from PDFs.
  why: Financial and scientific questions often live in tables.
  source: FinanceBench error analysis
- title: Streaming answers
  status: planned
  summary: Stream the answer while citations are verified at the end.
  why: Faster perceived response.
- title: Vector database connectors
  status: exploring
  summary: Use Pinecone, pgvector, Qdrant or similar as the retriever.
  why: Teams already have indexes.
- title: Scanned pages and images
  status: exploring
  summary: OCR and vision models for pages without a text layer.
  why: Many real documents are scans.
- title: PII redaction
  status: exploring
  summary: Mask personal data before text leaves your machine.
  why: Compliance.
- title: TypeScript SDK
  status: exploring
  summary: The same page-first workflow for JavaScript and TypeScript apps.
  why: Reach beyond Python.
- title: Hosted evaluation dashboard
  status: exploring
  summary: A shared place to compare pipelines and track results over time.
  why: Teams want history, not one-off runs.
- title: Query expansion
  status: shipping
  summary: The model writes alternative search phrasings and their rankings are fused with the original query's (Foveator(expand=2)).
  why: Questions and documents often use different words for the same thing.
  source: Uber enhanced agentic RAG (query optimiser); measured in the benchmarks
- title: Coverage-aware selection
  status: planned
  summary: Choose pages that cover complementary regions of a document instead of near-duplicates.
  why: Compression research finds redundant coverage wastes the budget.
  source: ComprExIT (arXiv:2602.03784)
- title: Structured history summaries
  status: planned
  summary: Summaries with fixed sections (goal, decisions, facts, open items) that keep recent turns verbatim.
  why: Partial, structured rewriting was consistently helpful across models.
  source: Beyond Token Savings (arXiv:2609.32961)
- title: Consistent answers across reworded questions
  status: building
  summary: 'Make the same question asked three ways return the same verdict: stabilise page selection across paraphrases and make the retry round deterministic.'
  why: In the first run Foveate agreed with itself on 57% of paraphrases and runs against 70% for plain retrieval.
  source: Benchmarks run 1 and 2
- title: Short facts inside long pages
  status: building
  summary: Score chunks as well as pages, so a single sentence inside a long page can still win a full-page slot.
  why: Plain chunk retrieval found hidden sentences at mid-document depths that Foveate's page ranking missed (100% against 50%).
  source: Benchmark run 2, needle items
- title: Fewer tokens for the same answer
  status: planned
  summary: Trim the condensed and outline tiers when the evidence is already strong, and stop after one round when the first answer is verified.
  why: Foveate sent about 45% more tokens than plain retrieval and took about 66% longer in run 2.
  source: Benchmark run 2
- title: Does the quote support the answer?
  status: planned
  summary: 'Go beyond checking that a quote exists on the page: check that the quote supports the claim, and report how strongly.'
  why: A real sentence can be quoted for a claim it does not prove.
  source: Context sufficiency work such as Provence
- title: Reasoning across pages and non-literal questions
  status: exploring
  summary: Help models connect a question to a fact that shares none of its words, with retrieval that understands the link and a controlled inference step.
  why: No pipeline solved NoLiMa-style needles (0 to 2%), including sending the whole text.
  source: NoLiMa (arXiv:2502.05167)
- title: More answer models and larger samples
  status: planned
  summary: Repeat the benchmarks with three or more answer models, 100 or more questions, and model-based compression methods such as ppa.
  why: Thirty questions and one model cannot separate small differences; rankings change between models.
  source: Beyond Token Savings (arXiv:2609.32961)
---

Status of each item. The list is the same one shown on the site's Coming next page. Items under "Being built now" and "Planned" that name a benchmark are the gaps the benchmarks found; they are listed here so the results are not only a score but a to-do list.

## Shipping in 0.2.0

* **Page-level selection and foveation.** Load PDF, DOCX, HTML, Markdown or text, rank pages with BM25, embeddings or both, and send each page at the detail it deserves.
* **Verified citations and abstention.** Every answer cites document, page and quote. Quotes are checked against the page; unsupported answers are retried, flagged or turned into not-found.
* **Agent page tools.** read_pages, search_document and document_outline for Pydantic AI, Google ADK and LangGraph.
* **History and tool-output compression.** Compress chat history with ppa, ushape, window and more. Shrink JSON, CSV, HTML and logs without a model.
* **Position-aware page order.** Put the strongest pages at the start and end of the evidence block and the weakest in the middle.
* **Query-aware condensing.** Condense neighbouring pages by keeping the sentences most relevant to the question, not the most frequent words.
* **Cache-friendly prompts.** Order prompts so the stable part comes first and report cached tokens, so provider prompt caches hit.
* **Tool-result clearing.** Replace old, re-fetchable tool results with a stub that keeps the call and its arguments.
* **Needle-in-a-haystack benchmark.** Length-by-depth grids with NoLiMa, a RULER-style generator and a LongBench v2 sample, for all four pipelines.
* **Compression benchmark.** Accuracy against compression ratio for each model-free method on documents, chat history, tool output and HotpotQA, with fact-retention metrics. Model-based methods are next.
* **Long-term memory.** Working notes per session and consolidated facts across sessions, with recall by query.
* **Evaluate your own documents in minutes.** Build a starter gold set from your own files: answerable, unanswerable and needle questions.
* **Query expansion.** The model writes alternative search phrasings and their rankings are fused with the original query's (Foveator(expand=2)).

## Being built now

* **Consistent answers across reworded questions.** Make the same question asked three ways return the same verdict: stabilise page selection across paraphrases and make the retry round deterministic.
* **Short facts inside long pages.** Score chunks as well as pages, so a single sentence inside a long page can still win a full-page slot.

## Planned

* **Context awareness.** Tell the model how many tokens it has used and how many remain.
* **Compression guideline tuning.** Learn compression instructions from failures: compare runs where full context succeeded and compressed context failed, and rewrite the guideline.
* **Table-aware PDF extraction.** Keep rows and columns when reading tables from PDFs.
* **Streaming answers.** Stream the answer while citations are verified at the end.
* **Coverage-aware selection.** Choose pages that cover complementary regions of a document instead of near-duplicates.
* **Structured history summaries.** Summaries with fixed sections (goal, decisions, facts, open items) that keep recent turns verbatim.
* **Fewer tokens for the same answer.** Trim the condensed and outline tiers when the evidence is already strong, and stop after one round when the first answer is verified.
* **Does the quote support the answer?.** Go beyond checking that a quote exists on the page: check that the quote supports the claim, and report how strongly.
* **More answer models and larger samples.** Repeat the benchmarks with three or more answer models, 100 or more questions, and model-based compression methods such as ppa.

## Exploring

* **Vector database connectors.** Use Pinecone, pgvector, Qdrant or similar as the retriever.
* **Scanned pages and images.** OCR and vision models for pages without a text layer.
* **PII redaction.** Mask personal data before text leaves your machine.
* **TypeScript SDK.** The same page-first workflow for JavaScript and TypeScript apps.
* **Hosted evaluation dashboard.** A shared place to compare pipelines and track results over time.
* **Reasoning across pages and non-literal questions.** Help models connect a question to a fact that shares none of its words, with retrieval that understands the link and a controlled inference step.

Open an issue to vote for an item or to propose another.
