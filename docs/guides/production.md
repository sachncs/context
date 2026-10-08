# Production

Foveate is async-first. Inside a service, `await foveator.aask(...)` and `await context.acompress(...)`. The
synchronous methods are for scripts and notebooks; they run the same code.

## Runtime and credentials

Build one `Runtime` per process and share it across tasks: it owns the cache
connection, the circuit-breaker state and the concurrency limits, which only
work as intended when shared. Caches and stores lock internally; the circuit
breaker assumes one event loop per process. `Runtime.from_env()` reads `FOVEATE_BACKEND`,
`FOVEATE_MODEL`, `FOVEATE_BASE_URL`, `FOVEATE_OPTIONS`, `FOVEATE_CACHE_DIR`, `FOVEATE_TIMEOUT_SECONDS`, `FOVEATE_RETRY_ATTEMPTS`,
`FOVEATE_CONCURRENCY`, and optionally `FOVEATE_RATE_LIMIT_PER_SECOND` and
`FOVEATE_DEADLINE_SECONDS`; malformed values raise `ConfigError` instead of being
ignored. Provider credentials come from the provider's own variables. Never
put keys in source, prompts, caches, notes or OKF bundles.

Close it on shutdown: `with Runtime.from_env() as runtime:` or `await
runtime.aclose()`.

## Reasoning models

Reasoning models can exhaust the token cap on hidden thinking and return no
visible text (`finish_reason="length"`). foveate retries with a 4x larger cap (3
times) and treats a reply with under half the cap visible as exhausted. To cut
cost and latency set a low effort for every call, e.g.
`FOVEATE_OPTIONS='{"reasoning_effort": "low"}'`; options are part of the cache
key.

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Timeout, 5xx, 429, connection error | Retried with exponential backoff and full jitter (`Retry-After` honoured), up to `FOVEATE_RETRY_ATTEMPTS` and the optional `FOVEATE_DEADLINE_SECONDS` budget |
| Auth, bad request, unknown error | Never retried; `PermanentBackendError` |
| Repeated failures | Circuit breaker opens, calls fail fast with `CircuitOpenError`, half-opens after a cool-down |
| Empty model output | One regeneration, then `ValidationError` |
| A leaf fails mid-compression | Siblings are cancelled, `CompressionError(step=..., cause=...)` is raised; leaves that finished are cached, so a retry resumes |
| Budget unreachable | `BudgetExceededError`, or hard truncation when `Overflow.TRUNCATE` |
| LLM down | Use `"ppa\|extractive"` to degrade to offline compression |
| Corrupt cache file | Quarantined as `*.corrupt` and recreated; bad rows are misses |

Catch `FoveateError` at the service boundary; `BackendError`,
`CompressionError`, `BudgetExceededError`, `ValidationError` and `ConfigError`
are its subclasses.

## Observability

The library logs to the `foveate` logger hierarchy with only a `NullHandler`
installed; configure handlers in your application. Pass observers to
`Runtime(observers=(...))`: `LoggingObserver` writes events, `MetricsObserver`
aggregates cache hits, backend calls, retries, tokens and latency. Every
compressed `Context` carries a `CompressionReport` with per-step token counts,
provider usage and estimated cost (supply a `PriceTable`). Do not log prompts
or completions by default.

## Storage and privacy

- Put `FOVEATE_CACHE_DIR`, notes and OKF bundles on private storage with least
  privilege; they can contain original or derived user content. Cache entries
  are plaintext. Use `SqliteCache(ttl_seconds=...)` and `max_entries` to bound
  retention and size.
- foveate sends no telemetry.
- OKF bundle reads refuse symlinks that resolve outside the bundle; writes are
  staged and swapped atomically so readers never see a partial bundle.
- Notes writes are atomic and guarded by a lock file for cross-process
  safety; `NotesStore.evict_to(max_bytes)` evicts oldest unpinned notes.

## Capacity

Concurrency is bounded twice: per operation (`Runtime.concurrency`) and per
backend (`ResilientBackend(max_concurrency=...)`). `PartitionSummarizeCombine`
rejects inputs that split into more than `max_leaves` (default 512) partitions.
Token counts use tiktoken when installed (`[tokenize]`), otherwise a
4-chars-per-token estimate; budgets exclude per-message framing tokens, so
leave headroom against the model's hard context limit.

## Release checklist

1. `make check` and `make build` pass locally.
2. Update `CHANGELOG.md` and the version in `pyproject.toml` and
   `foveate/__init__.py` (a test keeps them equal).
3. Push a `vX.Y.Z` tag. The release workflow re-runs the full CI matrix
   (Linux, macOS, Windows × Python 3.10-3.13), builds, publishes `foveate` to
   PyPI via trusted publishing, then creates the GitHub release.
4. Configure the PyPI trusted publisher for the `Release` workflow and `pypi`
   environment before the first release.

## Answering documents in production

* `Foveator.plan(...)` before the call; gate on `plan.estimated_cost_usd`.
* Alert on `answer.grounded == False` and on the share of `abstained` answers;
  set `ungrounded="abstain"` if a wrong claim costs more than a missing one.
* Retrieved text is fenced and escaped (`foveate.fencing`); treat that as
  defence in depth, and use `fencing.suspicious(page_text)` to log hostile pages.
* Trace with `foveate.tracing.TracingObserver` (OpenTelemetry spans carrying
  `gen_ai.*` token attributes).
* Pin the model name, and set `context_window` when your provider serves less
  than the model supports.
* Re-run [your evaluation](evaluating.md) before changing models or budgets.
