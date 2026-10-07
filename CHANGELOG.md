# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - Unreleased

A ground-up, breaking rewrite. There are no compatibility shims; this lists
what changed so you can port, not how to keep old code running.

### Added

- `Context`: immutable conversation + `Runtime`, with `compress`/`acompress`,
  `verify`/`averify`, `save`/`load`. `compress` returns a new `Context` with a
  `CompressionReport` and **guarantees the budget** (or raises
  `BudgetExceededError`, or truncates when `Overflow.TRUNCATE`).
- Compression strategies: `ppa`, `hierarchical`, `ushape`, `window`,
  `truncate`, `extractive`, `offload`; composition with `a+b` (pipeline) and
  `a|b` (fallback). Max-min fair budget allocation across messages.
- `Runtime` (backend, cache, tokenizer, observers, price table) replacing the
  global backend singleton; `Runtime.from_env()` with strict parsing.
- Resilient backends: retries with jitter for transient errors only,
  `Retry-After`, per-call timeouts, an overall retry deadline, circuit breaker,
  concurrency caps, an optional rate limit, error classification;
  `ScriptedBackend` for tests. Tokenizer is chosen from the model name.
- Concurrent leaf summarisation, single-flight request de-duplication, usage
  and cost accounting, structured events and `MetricsObserver`.
- SQLite cache with TTL, size bound and corruption recovery; cache keys cover
  the full request and prompt/strategy versions.
- OKF persistence as `Context.save/load` (plus `json`), atomic bundle writes.
- `FilesystemNotesStore` with persisted tags/timestamps, atomic writes and
  cross-process locking.
- `ceng.evolution` (ACE) with typed curator operations, checkpoint/resume and
  per-step error isolation; `ceng.bench` with a real playbook-injecting arm.
- Strict typing (`mypy --strict`), Google-style docstrings, a no-underscore
  naming gate, property-based tests, a 90% coverage gate, CI on Linux, macOS
  and Windows.

### Fixed (defects in 1.x)

- `budget_tokens` was never enforced.
- The benchmark "ceng" arm contained no playbook (it measured the baseline
  against itself); benchmark fixtures were missing from the wheel.
- Evolver cache keys ignored the question/playbook and curator bullet ids
  collided across steps; `curator_frequency` only gated trimming; reflection was
  gated on string equality and helpful/harmful tags were only updated on
  failures.
- Compaction reported zero token counts and its cache key ignored the system
  prompt and sampling options.
- `"50%"` parsed as 1.0; integer `0` answers were dropped; Formula parsing
  ignored thousands separators and `%`.
- Note tags and timestamps were never persisted; note writes were not atomic.
- Partitioning flattened newlines; SQLite connections were never closed.

### Removed

- `ppa_compress`, `compress_to_bundle`, `ppa_compress_to_okf`,
  `compact_messages`, `ppa_check`, `NotesManager`, `ceng.notes`,
  `ceng.compact`, `ceng.check`, `ceng.presets`, `ceng.playbooks`, `ceng.eval`,
  `ceng.tokens`, `ceng.log`, `ceng.compress`, and the `ceng-bench` command
  (v2 ships no command line; use the Python API).
- The global backend registry functions (`set_backend`, `get_backend`, ...).
- AppWorld stub, the `index_only` and `fallback_to_last` options, and the
  `litellm` hard dependency (now the `litellm` extra).
- Benchmark numbers published with 1.x (invalid; see `BENCHMARKS.md`).

Release history before 2.0 is available in the git log.
