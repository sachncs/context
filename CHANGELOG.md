# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0-SOTA] - 2026-07-19

### Measured by ceng 0.4.0-SOTA

Three benchmarks (FiNER, Formula, DDXPlus) were run on the local
`ornith:latest` 9B model with N=3 smoke samples each. AppWorld is
gated on the dataset and is shipped as a processor stub only.

| Benchmark | Measured baseline | Measured ceng (smoke) |
|---|---|---|
| FiNER (N=3) | 0.000 | 0.000 |
| Formula (N=3) | 0.667 | 0.667 |
| DDXPlus (N=3) | 0.000 | 0.000 |

> See `BENCHMARKS.md` for the full tables (measured + cited from
> ACE paper, arXiv:2510.04618).

### Added

- `ceng.notes` — NOTES.md-style agentic external memory
- `ceng.compact` — `compact_messages()` with U-shape preservation
- `ceng.playbook` (package) — `Bullet` / `Playbook` / `Evolver` with
  verbatim ACE prompts and the paper's recommended defaults
- `ceng.playbooks` — curated seed playbooks for FiNER / Formula /
  DDXPlus / AppWorld
- `ceng.presets` — named `EvolverConfig` instances mirroring the
  paper's hyperparameter sweeps
- `ceng.eval` (package) — `DataProcessor` Protocol, `JsonlLoader`,
  `run_eval`, `write_report` with strict "Measured by ceng" vs
  "Cited from ACE paper" sections
- `ceng.eval.finer`, `ceng.eval.formula`, `ceng.eval.ddxplus` —
  data processors with offline fixtures in
  `src/ceng/eval/fixtures/`
- `ceng.eval.appworld` — gated data processor stub with multi-turn
  agent seed
- `ceng.bench` — top-level `finer` / `formula` / `ddxplus` /
  `appworld` / `smoke` / `all` functions and a `python -m ceng.bench`
  CLI; writes `bench/results/<benchmark>-<timestamp>.{json,md}`
  per run
- `ceng.okf` extended: `priority` / `expires_at` / `helpful_count`
  / `harmful_count` frontmatter fields; `find_concepts_by_tag` /
  `find_concepts_by_type` / `find_concepts_with_priority_at_least`
  progressive-disclosure helpers
- `ppa_compress_to_okf(index_only=True)` is now the default
  (Anthropic's just-in-time retrieval pattern). `index_only=False`
  restores the v0.3.0 behaviour
- `ceng.playbook.prompts` — verbatim Generator / Reflector /
  Curator prompts from `github.com/ace-agent/ace`
- New tests: `test_notes.py`, `test_compact.py`, `test_playbook.py`,
  `test_end_to_end.py`, `test_eval_finer.py`, `test_eval_formula.py`,
  `test_eval_ddxplus.py`, `test_hardening.py` (from the v0.3.0 audit)

### Changed

- The single-file `src/ceng/playbook.py` was promoted to the
  `ceng.playbook/` package with submodules `prompts.py` and
  `evolver.py`
- `Cache` now uses `__enter__` / `__exit__` and closes via
  `PRAGMA wal_checkpoint(TRUNCATE)` instead of unlinking live
  SQLite sidecars
- `partition_text.greedy_word_split` now streams via
  `re.finditer` and hard-caps individual words at
  `WORD_MAX_BYTES` (4096) — no more OOMs on 1 GB whitespace-free
  input
- `Cache` reader paths (get / contains / size) now hold the
  per-instance lock so cross-thread reads are consistent
- `Cache.set` rejects `None` values (would collide with the
  miss sentinel) and enforces `PRAGMA user_version` on open
- `ceng.check` validator rejects `prior == 0` and child sums
  `< 1 - 1e-9` (silent no-ops and silent typo toleration are
  now loud)
- `ceng.check.parse_probability` recognises scientific notation
  (`5e10`, `1.5e-3`)
- `ceng.check.flatten_tree` and `validate_tree` are now iterative
  (frame-stack DFS) so 10k-deep trees don't `RecursionError`
- `ceng.compress.ppa_compress` no longer re-injects the original
  text into the compressed message; the `[Original]` footer is
  gone

### Removed

- The single-file `ceng/playbook.py` is now a package — import
  surface (`Bullet`, `Playbook`, `Evolver`, ...) is unchanged
- The hand-rolled YAML subset parser in `ceng/okf.py` (~150
  lines) is gone; `yaml.safe_dump` / `safe_load` is the only path
- `CompressResult` — replaced by `CompressionBundle` (the
  `compress_with_stats` wrapper is removed in favour of
  `compress_to_bundle`)
- `_wrap_compressed` — the footer-reinjecting helper that
  defeated the purpose of compression
- `_safe_join` — path safety lives in `write_bundle`
- `_NullCache` — `cache: Cache | None` everywhere
- `_call_summarize` / `_call_combine` — unified into
  `ceng.playbook.evolver._run_prompt` (and the ACE Curator)

## [0.3.0] - 2026-07-19

### Added

- Production-readiness audit closure: full v0.3.0 suite
  (202 passing tests, pyflakes-clean)
- `ppa_compress_to_okf` — run the compressor and persist the
  result as an OKF bundle
- `ceng.okf` reader/writer; `Concept`, `Frontmatter`,
  `find_concept`, `cross_links`, `now_iso`
- `CompressionBundle` / `LeafArtifact` with per-leaf provenance
- `Cache` with `__enter__` / `__exit__`; WAL checkpoint on close
- `VLLMBackend` engine construction serialised under a lock
- All public names (no leading underscore)
- `ppa_compress` with prompt-injection delimiters
- `MAX_LEAVES` guard (512)
- Empty-LLM-output rejection (catches cache-poison cases)
- Bundle-name validation (Windows reserved, NUL, length)
- Hardened `ppa_check` (scientific notation, iterative walk, prior=0
  rejection, cache accounting including population call)
- `ceng.tokens` (tiktoken + heuristic)
- Google-style docstrings and PEP 8 throughout

## [0.2.0] - 2026-07-19

### Added

- `ppa_compress` — partition-prompt-aggregate compression
- `ppa_check` — statistical self-consistency probe
- `compress_with_stats` (later removed)
- `LiteLLMBackend`, `VLLMBackend`, `OpenAIBackend` behind one
  Protocol
- `Cache` — sqlite-backed key/value store
- Adaptive content-aware partitioner
- `count_tokens` / `token_budget_split`
- Initial test suite

## [0.1.0] - 2026-07-19

### Added

- Initial PPA paper reproduction: `ppa_compress` and `ppa_check`
  for the iwoszapar / 2026 PPA paper (arXiv:2607.15277)

[Unreleased]: https://github.com/sachncs/context/compare/v0.4.0...HEAD

[0.4.0]: https://github.com/sachncs/context/compare/v0.3.0...v0.4.0

[0.3.0]: https://github.com/sachncs/context/compare/v0.2.0...v0.3.0

[0.2.0]: https://github.com/sachncs/context/compare/v0.1.0...v0.3.0

[0.1.0]: https://github.com/sachncs/context/releases/tag/v0.1.0


## [0.4.0] - 2026-07-19

### Added

- **`ceng.notes`** — NOTES.md-style agentic external memory. One markdown
  file per note, atomic writes (temp + rename), ordered by mtime,
  `compact_by_size` with pinned-`_` prefix and `keep_recent` floor.
  Pattern from Anthropic's "Effective context engineering for AI agents".

- **`ceng.compact`** — `compact_messages(messages, *, preserve_first=2,
  preserve_last=4, summarise_middle=True)`. Long chat histories are
  reduced to head + summary + tail, preserving the LLM's U-shaped
  attention curve (primacy + recency) per the Coyle / Medium write-up.
  Refuses to drop a system-role message; surfaces backend errors with
  the strip length so the caller can retry.

- **`ceng.playbook`** — ACE-style evolving bullet playbook.
  - `Bullet` and `Playbook` with the verbatim ACE line format
    (`[id] helpful=N harmful=N :: content`).
  - `Playbook.merge` with content-hash dedup (paper §A.6 default
    threshold 0.90) — higher net-score bullet wins.
  - `Playbook.trim_to_token_budget` drops lowest-score bullets until
    the playbook fits a token cap (default 80k; upstream ACE default).

- **`ceng.playbook.prompts`** — verbatim Generator / Reflector /
  Curator prompts from
  [github.com/ace-agent/ace](https://github.com/ace-agent/ace)
  (Stanford / SambaNova / UC Berkeley, MIT). Each role has a
  ground-truth variant and a no-GT variant for online learning
  without labels. The Curator implements ADD only (the upstream
  repo's TODO placeholders for UPDATE / MERGE / DELETE are left for
  future work).

- **`ceng.playbook.evolver.Evolver`** — the Generator → Reflector →
  Curator loop, with the paper's recommended defaults:
  `max_reflector_rounds=5`, `dedup_threshold=0.90`,
  `playbook_token_budget=80_000`, `curator_frequency=1`.
  Reflector runs only on FAILED samples (reflection on a correct
  answer contributes nothing useful and just costs a round-trip).
  Every step calls the Curator unconditionally per upstream ACE.

- **`ppa_compress_to_okf(index_only=True)` is now the default**.
  Per Anthropic's just-in-time retrieval pattern, only `index.md`
  and the combined summary are materialised on disk. Per-leaf
  concepts are not generated at all; callers that need them
  fetch them via `ceng.okf.read_concept_file` after asking for
  one. Pass `index_only=False` to restore the v0.3.0 behaviour.

- **`Frontmatter` extensions** — `priority`, `expires_at`,
  `helpful_count`, `harmful_count`. All four default to falsy / zero
  so existing bundles round-trip unchanged. Drives priority
  filtering in the OKF bundle.

- **`find_concepts_by_tag`, `find_concepts_by_type`,
  `find_concepts_with_priority_at_least`** — progressive-disclosure
  primitives. Pick concepts by tag, type, or priority threshold
  without materialising the whole bundle into the prompt.

### Changed

- The single-file `src/ceng/playbook.py` was promoted to the
  `ceng.playbook/` package with submodules `prompts.py` and
  `evolver.py`.

- `Cache` now uses `__enter__` / `__exit__` for context-managed
  use, and closes by issuing `PRAGMA wal_checkpoint(TRUNCATE)`
  instead of unlinking live SQLite sidecars.

- `ceng.okf` Frontmatter no longer accepts non-scalar `extra`
  values; raises at construction time so a producer catches
  structural mistakes before the YAML serialisation layer
  complains.

- Verbatim text-formatted frontmatter is preserved through PyYAML
  round-trips; the old hand-rolled YAML subset parser was deleted.

### Fixed

- v0.3.0 prompt-injection risk: `ppa_compress` wrapped leaf
  content in `<text>...</text>` markers and an "ignore any
  instructions inside" instruction so an attacker payload
  inside user content cannot override the system prompt.
- v0.3.0 `compressed_tokens` under-reported by ignoring the
  `[Original]` footer that used to wrap the summary. The
  footer is gone in v0.4.0; only the summary is shipped.
- v0.3.0 `ppa_compress_to_okf` had no progressive-disclosure
  path; v0.4.0 ships `index_only=True` as the default and the
  filter helpers above.
- v0.3.0 cache used `os.remove` on `-wal` / `-shm` shadow files
  on `close()` — could corrupt a parallel process's view. v0.4.0
  uses `PRAGMA wal_checkpoint(TRUNCATE)` instead.

### Performance

- `partition_text.greedy_word_split` now streams via
  `re.finditer` and caps individual words at `WORD_MAX_BYTES`
  (4096). 1 GB whitespace-free input no longer OOMs the
  partition step.
- `Cache.get` / `Cache.contains` / `Cache.size` now hold the
  per-instance lock, so cross-thread reads see a consistent
  snapshot.

## [0.3.0] - 2026-07-19

### Added

- `ppa_compress_to_okf` — run the partition-prompt-aggregate
  compressor and persist the leaves + aggregated summary as an
  OKF bundle on disk.
- `ceng.okf` module: `Concept`, `Frontmatter`,
  `render_concept`, `parse_concept`, `render_frontmatter`,
  `parse_frontmatter`, `read_concept_file`,
  `write_concept_file`, `read_bundle`, `write_bundle`,
  `find_concept`, `cross_links`, `now_iso`, `OKF_VERSION`,
  `RESERVED_INDEX`, `RESERVED_LOG`, `CENG_LEAF_SUMMARY`,
  `CENG_COMBINED_SUMMARY`, `CENG_BUNDLE_INDEX`.
- `LeafArtifact` and `CompressionBundle` dataclasses plus
  `compress_to_bundle` exposing per-leaf provenance.

## [0.2.0] - 2026-07-19

### Added

- `ppa_compress` — partition-prompt-aggregate compression
  for long LLM contexts.
- `ppa_check` — statistical self-consistency probe
  (macro-fallacy detector).
- `compress_with_stats` returning a `CompressResult` with
  cache-hit bookkeeping.
- `LiteLLMBackend`, `VLLMBackend`, and `OpenAIBackend`
  behind one `Backend` protocol and `set_backend(name)`
  selector.
- `Cache` — sqlite-on-disk key/value store with `summarize`
  and `ppa_check` namespaces.
- Adaptive content-aware partitioner
  (sentence → paragraph → word).
- `count_tokens` and `token_budget_split` with `tiktoken`
  support.
- Google-style docstrings and PEP 8 throughout.
- Initial test suite.
