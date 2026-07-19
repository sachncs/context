# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-07-19

### Added

- `ppa_compress` — partition-prompt-aggregate compression for long LLM contexts. Splits the longest user message, summarises each leaf (cached), and combines the leaf summaries into one coherent compressed message.
- `ppa_check` — statistical self-consistency probe (macro-fallacy detector). Asks a probability question at the population level and at every leaf of a user-supplied binary partition tree, then compares the direct estimate against the prior-weighted aggregate.
- `compress_with_stats` returning a `CompressResult` carrying `original_tokens`, `compressed_tokens`, `leaf_count`, `cache_hits`, and `cache_misses`.
- `LiteLLMBackend`, `VLLMBackend`, and `OpenAIBackend` behind one `Backend` protocol and a `set_backend(name)` selector. Heavy SDKs are imported lazily inside `complete()`.
- `Cache` — sqlite-on-disk key/value store with `summarize` and `ppa_check` namespaces, WAL journal mode, and thread-safe writes.
- Adaptive content-aware partitioner (sentence → paragraph → word) with token-bounded leaves.
- `count_tokens` and `token_budget_split` with `tiktoken` support and a `len // 4` heuristic fallback.
- Google-style docstrings and PEP 8 throughout.
- Test suite covering the cache, backend selector, lazy imports, partitioner, compressor (single and multi-leaf paths, cache reuse, kw forwarding, message immutability), consistency probe (macro-fallacy detection, parsing, tree validation, cache reuse), and the top-level package surface.
