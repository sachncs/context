# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First release.

### Long documents

- `Document`: page-addressable documents with loaders for PDF, DOCX, HTML,
  Markdown and text; `select("10-14,40")`, `around`, `page`, `outline`.
- Selection: BM25 (no dependencies), embeddings (any OpenAI-compatible
  endpoint), hybrid rank fusion, optional model re-rank.
- Foveation: full / condensed / outline / dropped page tiers under a token
  budget; model context-window registry for automatic budgets.
- `Foveator`: grounded answers with `(document, page, quote)` citations
  verified against the source, retry with feedback, flag or abstain;
  `plan()` dry run with token and cost estimate.
- Prompt-injection fencing for retrieved text.

- Optional query expansion (`Foveator(expand=2)`): the model writes alternative
  search phrasings and their rankings are fused with the original query's.
- Position-aware page order (`FoveationConfig(order="edges")`) and query-aware
  condensing (`query_aware=True`), both off by default.
- `Answer.support` and `Answer.needs_more_context`; `Usage.cached_tokens`.

### Agents

- `read_pages`, `search_document` and `document_outline` tools for Pydantic AI,
  Google ADK and LangGraph.
- Chat-history compression hooks for the same three frameworks.
- Tool-output reducers for JSON, CSV, HTML and logs (`tool_output`) and
  `clear_tool_results` for old tool output.
- `foveate.memory.Memory`: session notes, durable facts, recall and consolidation.

### Compression

- `Context(...).compress(method, budget=...)` returns a new `Context` with a
  `CompressionReport`; the budget is guaranteed or an error is raised.
- Strategies `ppa`, `hierarchical`, `ushape`, `window`, `truncate`,
  `extractive`, `offload`, `tool_output`; `a+b` pipelines and `a|b` fallbacks.

### Reliability and operations

- `Runtime` with cache, single-flight, truncation-aware retries for reasoning
  models and inline `<think>` stripping.
- `ResilientBackend`: retry with jitter and `Retry-After`, timeout, circuit
  breaker, concurrency and rate limits, total deadline.
- Backends: LiteLLM, OpenAI-compatible, in-process vLLM, none (LLM-free).
- Typed events with logging, metrics and OpenTelemetry observers.

### Evaluation

- Needle-in-a-haystack grids (`foveate.bench.needle`): literal, NoLiMa-style and
  multi-needle cases over any length and depth.
- Compression sweep (`foveate.bench.compression`): accuracy against compression
  ratio for each method on documents, chat history and tool output.
- Starter gold sets from your own documents (`starter_items`).

- Long-document benchmark (`foveate.bench.longdoc`): FinanceBench loader,
  gold-set builders (answerable, unanswerable, paraphrase, needle depth),
  pipelines `full-context`, `truncate`, `naive-rag`, `foveate`, metrics and
  report.
- Playbook evolution (ACE) with `finer`, `formula` and `ddxplus` benchmarks.
- Notes stores, portable OKF bundles, verification (`fits`, `macro_fallacy`).
