---
title: "Reliability: the runtime"
description: "How every model call goes through one path with caching, retries, timeouts, a circuit breaker and rate limits, and how reasoning models are handled."
---

Models fail in ordinary ways: a timeout, an overloaded server, a rate limit, an empty reply. A
library that sends prompts has to handle them the same way every time. In Foveate, every model call,
from compression to answering, goes through one method: `Runtime.complete`.

## What `Runtime` holds

A `Runtime` is created once per process and shared. It owns the backend, a response cache, the
tokenizer, observers, prices, provider options and (optionally) an embedder.

```python
from foveate import Runtime

with Runtime.from_env() as runtime:      # closes the cache and connections on exit
    ...
```

`from_env` reads these variables: `FOVEATE_BACKEND` (`openai` or `vllm`), `FOVEATE_MODEL`,
`FOVEATE_BASE_URL`, `FOVEATE_OPTIONS` (JSON), `FOVEATE_CONTEXT_WINDOW`, `FOVEATE_EMBEDDING_MODEL`,
`FOVEATE_CACHE_DIR`, `FOVEATE_TIMEOUT_SECONDS`, `FOVEATE_RETRY_ATTEMPTS`, `FOVEATE_CONCURRENCY`,
`FOVEATE_RATE_LIMIT_PER_SECOND` and `FOVEATE_DEADLINE_SECONDS`. Bad values raise `ConfigError`
instead of being ignored.

## What happens on every call

1. **Cache.** The request is fingerprinted (model, messages, temperature, token cap, options and the
   prompt version). A repeat is answered from the local cache without calling the provider.
2. **Single flight.** Identical requests in flight at the same time share one provider call.
3. **Backend.** The call goes through `ResilientBackend`:
   * **Retry** with exponential backoff and jitter on timeouts, 5xx and rate limits, honouring
     `Retry-After`. Bad requests and authentication errors are never retried.
   * **Timeout** per call and an optional **deadline** for the whole retry sequence.
   * **Circuit breaker** that opens after repeated failures, so a dead provider fails fast, then
     half-opens after a cool-down.
   * **Concurrency and rate limits** so you do not exceed your quota.
4. **Recovery.** Empty or cut-off replies are retried (next section).
5. **Accounting.** Tokens, cached tokens, latency and cost are emitted as events.

## Reasoning models

Reasoning models "think" before answering, and the thinking uses the same token allowance as the
answer. With a small cap they can spend all of it thinking and return nothing. Foveate detects a reply
cut off with less than half the cap visible, asks again with a cap four times larger (up to three
times), removes inline `<think>...</think>` blocks from the text, and caps the growth so prompt plus cap
never exceeds the model's window. If there is no room left, you get a clear `ValidationError` rather
than a request the server rejects.

Set `FOVEATE_OPTIONS='{"reasoning_effort": "low"}'` to cut thinking cost where the provider supports it.

## Backends

| Backend | Use |
|---|---|
| `OpenAIBackend` | Any OpenAI-compatible endpoint: OpenAI, vLLM, Ollama, NVIDIA, Together, gateways. Uses only the standard library, so Foveate needs no SDK. |
| `VLLMBackend` | An in-process vLLM engine (the `foveate[vllm]` extra, see [Install](../install.md), Linux) |
| `NoBackend` | `Runtime.without_llm()`: raises if called; for offline use and tests |

Write your own by subclassing `Backend` and implementing `complete`.

## Watching it

Observers receive typed events (`BackendCall`, `RetryScheduled`, `CacheLookup`, `StepFinished`).
`LoggingObserver` writes them, `MetricsObserver` totals calls, tokens, retries and cache hits, and
`foveate.tracing.TracingObserver` turns them into OpenTelemetry spans with `gen_ai.*` attributes.
Events carry sizes and model names, not prompt text.

## Next

[Safety](safety.md).
