---
title: "API overview"
description: "The public names of Foveate by module, with a one-line description of each."
---

Every public class and function has a docstring with arguments and the errors it raises; use
`help(foveate.Foveator)` or your editor. The package ships type information (`py.typed`) and is checked
with `mypy --strict`. Names that start with an underscore do not exist in Foveate: what is public is what
a package exports in `__all__`.

## Top level (`import foveate`)

| Name | Purpose |
|---|---|
| `Document` | Page-addressable document: `load`, `select`, `around`, `page`, `outline`, `text` |
| `Foveator` | Retrieve, foveate, answer, verify: `ask`, `aask`, `plan`, `index` |
| `Plan`, `Index` | The dry-run result; documents prepared for repeated questions |
| `Answer`, `Citation` | The grounded result and its verified evidence |
| `Context` | Messages plus runtime; `compress(method, budget=...)` |
| `Message`, `Role` | Chat message model |
| `Budget`, `Overflow`, `CompressionReport` | Token ceiling, overflow policy, what compression did |
| `Memory` | Session notes and durable facts |
| `Runtime` | Backend, cache, tokenizer, observers, prices, options, embedder |

## Modules

| Module | Contents |
|---|---|
| `foveate.documents` | `Document`, `Page`, `Loader` (and its registry), `PageChunker`, `parse_pages` |
| `foveate.selection` | `Retriever`, `BM25Retriever`, `EmbeddingRetriever`, `HybridRetriever`, `LlmReranker`, `build_retriever`, query expansion |
| `foveate.foveation` | `FoveationConfig`, `Tier`, `PagePlan`, `Foveation`, `allocate`, `arrange` |
| `foveate.assembly` | `Slot`, `allocate`: split a window between instructions, history and evidence |
| `foveate.models` | Context windows of known models, `auto_budget` |
| `foveate.grounding` | `Answer`, `Citation`, quote verification, `NOT_FOUND` |
| `foveate.fencing` | `escape`, `suspicious` |
| `foveate.compression` | `Compressor` registry; `Pipeline` (`a+b`), `Fallback` (`a\|b`) |
| `foveate.agents` | `DocumentTools`, `HistoryCompressor`, `HistoryAdapter` (framework-neutral) |
| `foveate.memory` | `Memory`: `remember`, `recall`, `message`, `consolidate`, `forget` |
| `foveate.backends` | `Backend`, `OpenAIBackend`, `VLLMBackend`, `NoBackend`, `ResilientBackend`, `RetryPolicy`, `CircuitBreaker`, embedders |
| `foveate.cache` | `Cache`, `MemoryCache`, `SqliteCache`, `NullCache` |
| `foveate.observability`, `foveate.tracing` | Events, `LoggingObserver`, `MetricsObserver`, `TracingObserver` (OpenTelemetry) |
| `foveate.stores`, `foveate.okf` | Notes stores; portable knowledge bundles |
| `foveate.verification` | `fits`, `macro_fallacy` verifiers for compressed contexts |
| `foveate.bench` | `longdoc` (long documents, gold sets, `starter_items`), `needle` (length x depth grids), `compression` (accuracy against ratio), `scoring` (exact match, F1, atom recall) |

## Compression methods

`ppa`, `hierarchical`, `ushape`, `window`, `extractive`, `selective`, `query`, `truncate`, `offload`,
`tool_output`, `clear_tool_results`. See [compression](../learn/compression.md).

More: [settings](settings.md), [errors](errors.md), [extending Foveate](extending.md).
