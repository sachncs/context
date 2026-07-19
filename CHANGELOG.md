# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-07-19

### Added

- `ppa_compress` — partition-prompt-aggregate compression for long LLM contexts.
- `ppa_check` — statistical self-consistency probe (macro-fallacy detector).
- `litellm`, `vllm`, and `openai` backend adapters behind one callable interface.
- sqlite-on-disk cache with two namespaces (`summarize`, `ppa_check`).
- Adaptive content-aware partitioner (sentence → paragraph → section).
- PEP8 / Google-style docstrings throughout.
- Test suite covering cache, backends, partitioner, compressor, and consistency check.
