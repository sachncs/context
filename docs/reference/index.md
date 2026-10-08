---
title: "Reference"
description: "Public names, modules, extension points and environment variables."
---

Public names are exported from the package or the module listed. Every public
class and function has a docstring with arguments and errors; use
`help(foveate.Foveator)` or your editor. The package ships type information
(`py.typed`) and is checked with `mypy --strict`.

## Top level (`import foveate`)

| Name | Purpose |
|---|---|
| `Document` | Page-addressable document: `load`, `select`, `around`, `page`, `outline`, `text` |
| `Foveator` | Retrieve, foveate, answer, verify: `ask`, `aask`, `plan`, `index` |
| `Plan`, `Index` | Dry-run result; prepared documents for repeated questions |
| `Answer`, `Citation` | Grounded result and its verified evidence |
| `Context` | Messages plus runtime; `compress(method, budget=...)` |
| `Message`, `Role` | Chat message model |
| `Budget`, `Overflow`, `CompressionReport` | Token ceiling, overflow policy, what compression did |
| `Runtime` | Backend, cache, tokenizer, observers, prices, options, embedder |

## Modules

| Module | Contents |
|---|---|
| `foveate.documents` | `Document`, `Page`, `Loader` (+ registry), `PageChunker`, `parse_pages` |
| `foveate.selection` | `Retriever`, `BM25Retriever`, `EmbeddingRetriever`, `HybridRetriever`, `LlmReranker`, `build_retriever` |
| `foveate.foveation` | `FoveationConfig`, `Tier`, `PagePlan`, `Foveation`, `allocate` |
| `foveate.assembly` | `Slot`, `allocate`: split a window between instructions, history, evidence |
| `foveate.models` | Context windows of known models, `auto_budget` |
| `foveate.grounding` | `Answer`, `Citation`, quote verification, `NOT_FOUND` |
| `foveate.fencing` | `escape`, `suspicious` |
| `foveate.compression` | `Compressor` registry: `ppa`, `hierarchical`, `ushape`, `window`, `extractive`, `selective`, `query`, `truncate`, `offload`, `tool_output`, `clear_tool_results`; `Pipeline` (`a+b`), `Fallback` (`a\|b`) |
| `foveate.backends` | `Backend`, `LiteLLMBackend`, `OpenAIBackend`, `VLLMBackend`, `NoBackend`, `ResilientBackend`, `RetryPolicy`, `CircuitBreaker`, embedders |
| `foveate.cache` | `Cache`, `MemoryCache`, `SqliteCache`, `NullCache` |
| `foveate.observability` | `Event`s, `Observer`, `LoggingObserver`, `MetricsObserver` |
| `foveate.tracing` | `TracingObserver` (OpenTelemetry) |
| `foveate.integrations` | `history`, `tools`, `pydantic_ai`, `adk`, `langgraph` |
| `foveate.stores`, `foveate.okf` | Notes stores and portable knowledge format for `offload` and evolution |
| `foveate.verification` | `fits`, `macro_fallacy` verifiers for compressed contexts |
| `foveate.evolution` | ACE-style playbook evolution |
| `foveate.bench` | `Benchmark`, `Runner` (playbook tasks); `bench.longdoc` (long documents, gold sets, `starter_items`); `bench.needle` (length x depth grids); `bench.compression` (accuracy against compression ratio); `bench.scoring` (exact match, F1, atom recall) |
| `foveate.memory` | `Memory`: session notes, durable facts, `recall`, `consolidate` |

## Extending

Every extension point is an abstract class with a registry, so adding one is a
subclass plus a decorator:

| To add | Subclass | Register with |
|---|---|---|
| A compression method | `compression.Compressor` | `@Compressor.register("name")` |
| A tool-output format | `compression.Reducer` | `@Reducer.registry.register("name")` |
| A document format | `documents.Loader` | `@Loader.registry.register("name")` |
| A retriever | `selection.Retriever` | `@Retriever.registry.register("name")` |
| A benchmark pipeline | `bench.longdoc.Pipeline` | `@Pipeline.registry.register("name")` |
| A model provider | `backends.Backend` | pass it to `Runtime(backend=...)` |
| A cache | `cache.Cache` | pass it to `Runtime(cache=...)` |
| An observer | `observability.Observer` | `Runtime(observers=(...))` |

## Environment variables

`FOVEATE_BACKEND` (litellm, openai, vllm), `FOVEATE_MODEL`, `FOVEATE_BASE_URL`,
`FOVEATE_OPTIONS` (JSON), `FOVEATE_CONTEXT_WINDOW`, `FOVEATE_EMBEDDING_MODEL`,
`FOVEATE_CACHE_DIR` (empty disables), `FOVEATE_TIMEOUT_SECONDS`,
`FOVEATE_RETRY_ATTEMPTS`, `FOVEATE_CONCURRENCY`, `FOVEATE_RATE_LIMIT_PER_SECOND`,
`FOVEATE_DEADLINE_SECONDS`, `FOVEATE_CACHE_HOME` (benchmark downloads).
