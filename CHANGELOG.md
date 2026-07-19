# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-07-19

### Added

- `ppa_compress_to_okf` — run the partition-prompt-aggregate
  compressor and persist the leaves + aggregated summary as an
  [Open Knowledge Format v0.1](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
  bundle on disk. Each concept is one markdown file with YAML
  frontmatter (`type`, `title`, `description`, `tags`, `timestamp`)
  and a top-level `index.md` cross-links everything.
- `ceng.okf` module: `Concept`, `Frontmatter`, `render_concept`,
  `parse_concept`, `render_frontmatter`, `parse_frontmatter`,
  `read_concept_file`, `write_concept_file`, `read_bundle`,
  `write_bundle`, `find_concept`, `cross_links`, `now_iso`,
  `OKF_VERSION`, `RESERVED_INDEX`, `RESERVED_LOG`,
  `CENG_LEAF_SUMMARY`, `CENG_COMBINED_SUMMARY`, `CENG_BUNDLE_INDEX`.
  Dependency-free YAML subset parser keeps the package
  zero-dep-friendly.
- `LeafArtifact` and `CompressionBundle` dataclasses plus
  `compress_to_bundle` exposing per-leaf provenance so the OKF
  exporter can run without re-summarising.
- `Concept.path` is normalised to `pathlib.Path` on construction;
  `write_bundle` accepts both string and `Path` paths.

## [0.1.0] - 2026-07-19

### Added

- `ppa_compress` — partition-prompt-aggregate compression for long LLM contexts.
- `ppa_check` — statistical self-consistency probe (macro-fallacy detector).
- `compress_with_stats` returning a `CompressResult` with cache-hit bookkeeping.
- `LiteLLMBackend`, `VLLMBackend`, and `OpenAIBackend` behind one `Backend` protocol and a `set_backend(name)` selector.
- `Cache` — sqlite-on-disk key/value store with `summarize` and `ppa_check` namespaces.
- Adaptive content-aware partitioner (sentence → paragraph → word).
- `count_tokens` and `token_budget_split` with `tiktoken` support.
- Google-style docstrings and PEP 8 throughout.
- Test suite covering cache, backends, partitioner, compressor, consistency probe, and the top-level package surface.
