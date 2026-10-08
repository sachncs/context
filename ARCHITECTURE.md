# Architecture

Foveate has two value types, `Document` (pages) and `Context` (messages), and
a set of extension points that are abstract base classes with explicit
registries. Documents flow through selection, foveation and grounding;
conversations and tool output flow through compression.

```
question + Document(s)
   -> selection (Retriever: bm25 | embedding | hybrid | rerank)  -> page scores
   -> foveation (allocate FULL / CONDENSED / OUTLINE / DROPPED under a budget)
   -> prompt (fenced pages + outline)  -> Runtime.complete -> JSON answer
   -> grounding (verify quotes on the cited pages) -> retry / flag / abstain
   -> Answer(text, citations, grounded, usage, cost)
```

```
Context (frozen)  --compress()-->  Context (frozen, with CompressionReport)
   |  messages: tuple[Message, ...]
   |  runtime:  Runtime  ---- backend, cache, tokenizer, observers, prices
   |  report, metadata
```

## Layers (dependencies point downward)

| Layer | Package / module | Role |
|---|---|---|
| API | `foveator`, `context`, `agents` | `Foveator.ask/plan`; `Context.compress`; framework-neutral agent tools and history compression |
| Documents | `documents`, `selection`, `foveation`, `assembly`, `grounding`, `models`, `fencing` | Pages, retrieval, fidelity tiers, slots, citation checks, model windows, injection fencing |
| Strategies | `compression`, `verification` | `Compressor`, `Verifier` ABCs and implementations |
| Domain | `partition`, `okf`, `stores`, `memory`, `evolution`, `bench` | Partitioners, OKF bundles, notes stores, agent memory, ACE playbook evolution, benchmarks |
| Services | `runtime` | `Runtime.complete`: cache, single-flight, validation, accounting |
| I/O | `backends`, `cache`, `tokenizers` | Providers + resilience, caches, token counting |
| Foundation | `messages`, `errors`, `usage`, `prompts`, `observability`, `tracing`, `internals` | Value types and helpers |

## Extension points

Every one is an ABC with a `Registry` owned by the class (never a module
global). Registering is a decorator; unknown names raise `ConfigError` listing
the known ones.

| ABC | Registry key examples | Implementations |
|---|---|---|
| `Compressor` | `ppa`, `hierarchical`, `ushape`, `window`, `truncate`, `extractive`, `offload`, `tool_output` | plus `Pipeline` (`a+b`) and `Fallback` (`a\|b`) |
| `Verifier` | `fits`, `macro_fallacy` | |
| `Loader` | `pdf`, `docx`, `html`, `markdown`, `text` | |
| `Retriever` | `bm25`, `embedding`, `hybrid`, `rerank` | |
| `Embedder` | | `HashingEmbedder`, `OpenAIEmbedder` |
| `Reducer` | `json`, `csv`, `html`, `log` | used by `tool_output` |
| `Benchmark` | `finer`, `formula`, `ddxplus` | playbook tasks graded deterministically |
| `Pipeline` (bench) | `full-context`, `truncate`, `naive-rag`, `foveate` | |
| `Backend` | `openai` (stdlib HTTP, any OpenAI-compatible server), `vllm`, `none` | wrapped by `ResilientBackend` |
| `Cache` | `memory`, `sqlite`, `null` | |
| `Tokenizer` | `heuristic`, `tiktoken` | |
| `Partitioner` | `recursive`, `fixed` | |
| `Codec` | `okf`, `json` | |
| `NotesStore` | `memory`, `filesystem` | |
| `CuratorOp` | `ADD`, `UPDATE`, `MERGE`, `DELETE` | |
| `Observer` | | `LoggingObserver`, `MetricsObserver`, `TracingObserver` |
| `HistoryAdapter` | | Pydantic AI, ADK, LangChain/LangGraph messages |

Strategies are **frozen dataclasses whose fields are their options**, so
options are validated at construction, hashable, and visible in `repr`.

## The compression contract

`Compressor.compress` is a template method:

1. If the context already fits, return it unchanged (no LLM calls).
2. Otherwise call the strategy's `run`.
3. Enforce the budget: raise `BudgetExceededError`, or hard-truncate when
   `Budget.overflow` is `TRUNCATE` (the report records `truncated=True`).
4. Attach a `CompressionReport` (tokens before/after, per-step records,
   provider usage, estimated cost, duration).

Strategies that need an LLM call `Compressor.ask`, which builds the request
from a `PromptTemplate`, routes it through `Runtime.complete`, and records a
`StepRecord`. Multi-message budgets use max-min fair allocation
(`compression.fit.fair_targets`).

## Runtime.complete: the only LLM call path

Every LLM call in the library (compression, verification, memory, evolution, bench)
goes through `Runtime.complete`:

1. Build a `Request` and a cache key from `namespace` + the **full request
   fingerprint** (model, rendered messages, temperature, max tokens). The
   namespace carries the strategy name/version and `PromptTemplate.fingerprint`,
   so editing a prompt or a strategy cannot serve stale answers.
2. Cache lookup (corrupt entries are treated as misses).
3. Single-flight: concurrent identical requests share one backend call.
4. Backend call (resilient: retry, timeout, breaker, concurrency cap, optional
   rate limit). Empty or truncated output is retried: a reply cut off by the
   token cap with less than half of it visible (hidden reasoning) is re-asked
   with a 4x larger cap, up to 3 times; plain empty text is re-asked once.
5. Cache write, usage and event emission.

## Async core, one sync bridge

All work is `async`. `Context.compress` is `run_sync(acompress(...))`;
`internals.runner.run_sync` uses `asyncio.run`, or a worker thread when a loop
is already running (Jupyter). Leaf summaries run concurrently through
`internals.concurrency.gather_bounded`, which cancels siblings on first failure.

## Model output length

Models routinely overshoot a requested length. `Compressor.ask` therefore
follows an over-long answer with up to two "rewrite shorter" calls: a 30%
tolerance for intermediate (leaf) summaries and a hard limit for the final
stage, so budgets are met by editing instead of cutting.

## Event loops

The sync API runs each call in a fresh event loop, but async clients and
primitives are bound to the loop that created them. `internals.looplocal`
gives every running loop its own provider client, semaphore and pacing lock
(held weakly), which prevents "Event loop is closed" errors on repeated calls.

## Agents without framework dependencies

`foveate.agents.HistoryCompressor` is framework-neutral: if the history fits, it is returned untouched;
otherwise it cuts at a turn boundary (never between a tool call and its result), flattens the older messages
to text, compresses them (`ushape` by default, with leading system messages kept) and falls back to an offline
method on provider failure. A `HistoryAdapter` maps one framework's messages to plain text and back.
`DocumentTools` gives any agent `read_pages`, `search_document` and `document_outline` as typed functions.

The adapters for Pydantic AI, Google ADK, LangGraph and Strands Agents are separate packages in the
`integrations/` directory (`foveate-pydantic-ai`, `foveate-adk`, `foveate-langgraph`, `foveate-strands`). The
core package imports none of those frameworks, so installing Foveate never pulls in an agent stack.

## No SDKs in the core

Model calls use `urllib` in a worker thread (`backends/http.py`), speaking the OpenAI chat-completions and
embeddings formats that vLLM, Ollama, NVIDIA, Together and most gateways also serve. HTTP statuses map onto the
transient/permanent error taxonomy (429 and `Retry-After`, 408, 409 and 5xx are retried; other 4xx are not).

## Naming rule: no underscore-prefixed names

Google's style guide uses a leading underscore for non-public names. This
project deliberately does not: `tests/test_style.py` AST-scans `foveate/` and
fails on any function, class, variable, argument or attribute that starts with
a single underscore (dunders are fine). The public API is delimited instead by:

1. an explicit `__all__` in every package `__init__` (checked by a test),
2. the `internals` package for helpers that are not API,
3. documentation.

Anything not exported from a package `__init__` or this document is not a
compatibility promise, even though Python cannot enforce that.

## Design notes

- `Cache` is synchronous: it wraps local SQLite/dicts, so async buys nothing.
- Playbook bullet ids are deterministic `uuid5` of normalised content (same
  content, same id), which keeps cache keys reproducible and cannot collide.
- No test double ships in the package: `ScriptedBackend` lives in
  `tests/faults.py` for fault injection only, and `Runtime.without_llm()`
  (`NoBackend`) serves LLM-free use.
- Cache keys are SHA-256 fingerprint strings (`internals.hashing`).
- Messages are plain text: multi-part content is flattened on ingest.
- `Observer` has one `handle(event)` method over typed `Event` subclasses.
- Citation checking is deterministic (normalised substring, then a
  longest-common-subsequence ratio), so it needs no model and cannot be talked
  out of its verdict by the document.
- Inline reasoning (`<think>...</think>`) is stripped in `Runtime.complete`,
  before caching and parsing.
