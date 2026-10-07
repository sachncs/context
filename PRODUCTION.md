# Production operations

The published distribution is `ceng-context`; the import name is `ceng`.

ceng is async-first. Inside a service, `await context.acompress(...)`. The
synchronous methods are for scripts and notebooks; they run the same code.

## Runtime and credentials

Build one `Runtime` per process and share it across tasks: it owns the cache
connection, the circuit-breaker state and the concurrency limits, which only
work as intended when shared. Caches and stores lock internally; the circuit
breaker assumes one event loop per process. `Runtime.from_env()` reads `CENG_BACKEND`,
`CENG_MODEL`, `CENG_CACHE_DIR`, `CENG_TIMEOUT_SECONDS`, `CENG_RETRY_ATTEMPTS`,
`CENG_CONCURRENCY`, and optionally `CENG_RATE_LIMIT_PER_SECOND` and
`CENG_DEADLINE_SECONDS`; malformed values raise `ConfigError` instead of being
ignored. Provider credentials come from the provider's own variables. Never
put keys in source, prompts, caches, notes or OKF bundles.

Close it on shutdown: `with Runtime.from_env() as runtime:` or `await
runtime.aclose()`.

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Timeout, 5xx, 429, connection error | Retried with exponential backoff and full jitter (`Retry-After` honoured), up to `CENG_RETRY_ATTEMPTS` and the optional `CENG_DEADLINE_SECONDS` budget |
| Auth, bad request, unknown error | Never retried; `PermanentBackendError` |
| Repeated failures | Circuit breaker opens, calls fail fast with `CircuitOpenError`, half-opens after a cool-down |
| Empty model output | One regeneration, then `ValidationError` |
| A leaf fails mid-compression | Siblings are cancelled, `CompressionError(step=..., cause=...)` is raised; leaves that finished are cached, so a retry resumes |
| Budget unreachable | `BudgetExceededError`, or hard truncation when `Overflow.TRUNCATE` |
| LLM down | Use `"ppa\|extractive"` to degrade to offline compression |
| Corrupt cache file | Quarantined as `*.corrupt` and recreated; bad rows are misses |

Catch `CengError` at the service boundary; `BackendError`,
`CompressionError`, `BudgetExceededError`, `ValidationError` and `ConfigError`
are its subclasses.

## Observability

The library logs to the `ceng` logger hierarchy with only a `NullHandler`
installed; configure handlers in your application. Pass observers to
`Runtime(observers=(...))`: `LoggingObserver` writes events, `MetricsObserver`
aggregates cache hits, backend calls, retries, tokens and latency. Every
compressed `Context` carries a `CompressionReport` with per-step token counts,
provider usage and estimated cost (supply a `PriceTable`). Do not log prompts
or completions by default.

## Storage and privacy

- Put `CENG_CACHE_DIR`, notes and OKF bundles on private storage with least
  privilege; they can contain original or derived user content. Cache entries
  are plaintext. Use `SqliteCache(ttl_seconds=...)` and `max_entries` to bound
  retention and size.
- ceng sends no telemetry.
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
   `src/ceng/__init__.py` (a test keeps them equal).
3. Push a `vX.Y.Z` tag. The release workflow re-runs the full CI matrix
   (Linux, macOS, Windows × Python 3.10-3.13), builds, publishes to PyPI via
   trusted publishing, then creates the GitHub release.
4. Configure the PyPI trusted publisher for the `Release` workflow and `pypi`
   environment before the first release.
