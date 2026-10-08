# Foveate: context engineering for LLM apps (rename, long-document pipeline, proof, docs)

## Context

The repo is a well-tested but narrow library (compress a chat). The owner's review:
"context engineering is not context compression", docs are stale/duplicated, the name `ceng`
is unusable on PyPI, `src/` is unwanted, and the product is "useless" unless it **proves** that
uploading a ~200-page document goes better with it than without it (quality, grounding,
reliability). Nothing was ever published to PyPI, so there are no users to keep compatible with.

Facts established while planning:
- PyPI `ceng` is **taken** (civil-engineering tool). **`foveate` is free** on PyPI and npm; GitHub hits are unrelated graphics projects.
- vLLM publishes **no macOS wheels** (Linux x86_64/aarch64 only); Docker via colima is installed but stopped; Mac is M3 Pro / 19 GB.
- **FinanceBench** (HF `PatronusAI/financebench`, CC BY-NC 4.0): 150 Q&A over 84 real SEC 10-K/10-Q PDFs
  (dozens to hundreds of pages), reference answers, evidence text + page numbers; PDFs fetchable from issuer sites.
- Live NVIDIA models are mostly reasoning models; `openai/gpt-oss-20b` + `reasoning_effort=low` is fast (0.5 s).
- Git: `origin` = github.com/sachncs/context; `gh` is logged in as `sachn-cs`. Branch `v2-rewrite` has 14 local commits, none pushed.

## Decisions (confirmed with the owner)

- **Brand/PyPI/import name: `foveate`** (fovea = sharp centre of vision; to *foveate* = keep full detail where it matters, coarse elsewhere). `pip install foveate`, `import foveate`.
- **Flat layout**: `src/ceng/` becomes top-level `foveate/`; no `src/`.
- **Retrieval**: BM25 default (zero deps) + optional embeddings via any OpenAI-compatible `/embeddings` + hybrid + optional LLM re-rank.
- **Benchmark data**: public **FinanceBench** (real filings) as the base; we build our own **gold sets** for grounding, unanswerable questions, reliability. Optional second public set (LongBench-v2, Apache-2.0) on a small sample.
- **vLLM**: Linux CPU in Docker (colima) running a small model (MiniCPM5-1B/2B; fall back to Qwen2.5-0.5B/SmolLM2 if vLLM lacks the architecture), exercised through both the OpenAI-compatible server and the in-process `VLLMBackend`.
- **CI**: delete the real-model/framework job and every mention of the `NVIDIA_API_KEY` secret. macOS and Windows stay in the unit matrix; prove them by pushing.
- **Git**: strip the `Co-Authored-By: Claude...` trailer from all commits (history rewrite of local-only commits), no trailer on new commits, force-push `v2-rewrite`, open a PR, merge after CI is green on all three OSes. Omit the "Generated with Claude Code" PR footer.
- Reset version to **0.1.0** (never published; drop all "v1 to v2 / removed" migration text).

## The product story (what the docs and API must answer)

"If I upload 200 pages, what happens?"
- **Without foveate**: it either exceeds the model window (provider error / silent truncation) or it fits but costs 100k+ tokens per question, is slow, suffers "lost in the middle", and returns uncited answers that cannot be audited.
- **With foveate**: `Document.load()` gives pages/outline/token counts; `Plan` (dry run) shows tokens, cost and what will be kept; relevant pages are selected (plus neighbours) at **full fidelity (fovea)**, adjacent material is compressed (**parafovea**), the rest is a cheap page outline (**periphery**) the model can ask to expand; the answer carries **page citations**, is **verified** against the cited text, and **abstains** when unsupported.
- "Specific pages": `doc.pages("12-14")`, `doc.around(40, radius=2)`, `doc.search("capex")`, or let an agent call `read_pages` / `search_document` tools.

## Phases

### 0. Housekeeping and git (first, mechanical)
- Rewrite history to drop Claude trailers (`git filter-branch --msg-filter` on `master..v2-rewrite`); verify `git log` has none.
- `git mv src/ceng foveate`; rename every `ceng` to `foveate`: imports, env vars `CENG_*` to `FOVEATE_*`, logger names, cache dirs `.ceng` to `.foveate`, test-style scan path, ruff/mypy/coverage paths, CI wheel checks, examples, site, docs.
- `pyproject.toml`: `name="foveate"`, `version="0.1.0"`, `packages.find` with `include=["foveate*"]` (so `tests/ examples/ site/` never ship), package-data paths, extras (`pdf`, `embeddings`, `frameworks`, `vllm`, `openai`, `litellm`, `tokenize`), classifiers "Beta", keywords for context engineering. Verify the wheel contains only `foveate/`.
- Delete stale files: `plan.md`, old `BENCHMARKS.md` text, CHANGELOG history, "removed in 2.0" lists, v1 references. Rewrite `CHANGELOG.md` as a fresh 0.1.0.
- Critical files: `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`, `tests/test_style.py`, `tests/test_package.py`, `tests/integration/conftest.py`.

### 1. Documents and page-level control (answers "200 pages / specific pages")
New package `foveate/documents/`:
- `Document` (frozen): `pages: tuple[Page, ...]` (`Page(number, text, tokens, headings)`), `outline`, metadata; `Document.load(path|bytes, format=None)` through a **`Loader` ABC + registry** (`pdf` via optional `pypdf`, `text`, `markdown`, `html`, `docx` optional); `doc.pages("10-14,40")`, `doc.around(page, radius)`, `doc.slice()`, `doc.token_count`, `doc.to_context(pages=...)`.
- Structure-aware chunking for retrieval units that keep page + heading provenance (`Chunker` ABC: page, paragraph/semantic, table-preserving).
- Corpus of several documents with stable ids (`doc_id`, `page`) used in citations.

### 2. Selection (what goes in the window)
New `foveate/selection/`: `Selector` ABC + registry with `PageRange`, `Neighbours`, `BM25` (pure Python), `Embedding` (OpenAI-compatible `/embeddings`, on-disk vector cache, reuses `cache/`), `Hybrid` (reciprocal-rank fusion), `LlmRerank`. Output `Selection(items, scores, reasons)`.
- `Foveation` allocator (the core idea): given scores + budget, assign each page a fidelity tier FULL / COMPRESSED / OUTLINE / DROPPED using existing `compression` strategies (`extractive`, `ushape`, `ppa`) and `compression.fit.fair_targets` for max-min fair budget.
- Model registry (`foveate/models.py`): context window + price per known model, so `budget="auto"` = window minus reserve; user override always wins.

### 3. Context assembly and grounded answering
New `foveate/assembly/` and `foveate/grounding/`:
- `Assembler`: named **slots** (instructions, memory, history, evidence, tools) with priorities and budgets; reuses `Context`, `Budget`, `Overflow`. Retrieved text is fenced as data with an injection-resistant template (extends `prompts.PromptTemplate`).
- `Foveator` (the high-level object, composition of selector + allocator + assembler + answerer): `Foveator(runtime, budget=...).ask(question, documents) -> Answer`.
- `Answer`: `text`, `citations: tuple[Citation(doc_id, page, quote)]`, `grounded: bool`, `support: float`, `abstained: bool`, `plan`, `report` (tokens, cost, steps).
- `Verifier` (new class in the existing `verification` ABC): `CitationVerifier` checks every cited quote exists on the cited page (string match, deterministic) and `SupportVerifier` asks the model whether the cited text supports each claim; unsupported answers are re-asked once with more evidence, else the answer abstains ("not found in the provided pages").
- `Plan` / `foveate.plan(...)`: dry run returning token counts, estimated cost and the tier assignment before any model call.

### 4. Agent tools (the other half of "specific pages")
`foveate/integrations/` gains `read_pages(doc, start, end)` and `search_document(query, k)` tool factories for Pydantic AI, Google ADK and LangGraph (reuses existing adapter modules and `HistoryAdapter`); keep history compression. Real-framework tests extend `tests/integration/test_frameworks.py`; **no-model adapter tests** (`tests/test_framework_adapters.py`, `importorskip`) cover flatten/rebuild/starts_turn with the real message classes so a new CI job can run them without a key (remove the `coverage omit` for the three modules).

### 5. Other forgotten/missing features (prioritised for "a million developers")
Tier A (build now, small, high value): tool-output compression for agents (JSON/HTML/log/CSV structure-aware `Compressor`s: prune keys, truncate arrays with counts, collapse repeated log lines); cost/token estimator and `Plan`; OpenTelemetry-compatible tracing via the existing `Observer` ABC; prompt-injection fencing; model/window registry.
Tier B (build if time after the benchmark): long-term conversation memory (extends `stores.NotesStore` with summaries + recall), table-aware PDF extraction, streaming `ask`.
Tier C (roadmap only, documented, not built): vector-DB connectors, multimodal pages (vision), PII redaction, TypeScript SDK, hosted eval dashboard.

### 6. Benchmark: with vs without foveate (the proof)
`foveate/bench/longdoc/` (extends existing `Benchmark`, `Runner`, `Arm`, `BenchResult`, and the report writer):
- **Data**: `FinanceBenchLoader` downloads the jsonl and PDFs at run time into `~/.cache/foveate/financebench` (CC BY-NC: never redistributed; only our derived gold annotations are committed under `foveate/bench/longdoc/gold/`).
- **Gold sets we build** (script `foveate/bench/longdoc/build_gold.py`, outputs reviewed and versioned): (a) page-level evidence labels from FinanceBench evidence pages (grounding); (b) **unanswerable** set: question paired with a filing whose text provably lacks the answer value (automatic text search + spot check); (c) **reliability** set: paraphrased questions (cached LLM paraphrase, reviewed) for consistency; (d) **position sweep**: a synthetic needle sentence inserted at 0/25/50/75/100% depth of a real filing (lost-in-the-middle).
- **Pipelines compared** (all via a `Pipeline` ABC): `FullContext` (whole document; records window overflow as a failure), `NaiveTruncate`, `NaiveRag` (fixed chunks, top-k, no foveation or verification: the honest competitor), `Foveate`.
- **Metrics**: answer correctness (deterministic numeric/string match first, LLM judge fallback with a rubric; a stronger judge model than the answerer if one is fast enough), citation page precision/recall vs gold, faithfulness (claim support judged against cited text), abstention accuracy on unanswerables, run-to-run agreement over 3 seeded runs and paraphrases, accuracy vs needle depth, plus prompt tokens, estimated cost, latency.
- **Runs**: primary `openai/gpt-oss-20b` (`reasoning_effort=low`) on a stratified sample (default 30 questions / ~20 filings, `--full` for all 150); a **small local model through vLLM** (window 4-32k, where the full-context baseline cannot run at all); responses cached so reruns are cheap. A `--dry-run` prints the token/cost estimate first.
- **Honesty gate**: results (including losses) are written to `docs/benchmarks.md` with date, model, n, seeds and exact commands. Targets used to decide whether to iterate, not to massage numbers: non-inferior accuracy to full-context within a stated margin at far fewer tokens; page-citation recall materially above naive RAG; correct abstention rate above both baselines; lower run-to-run variance. If foveate does not beat NaiveRag on something, we fix the pipeline (selector, tiers, verifier) before publishing and say so in the docs.

### 7. vLLM
- `docker/vllm-cpu/Dockerfile` + `docker/vllm-cpu/README`: start colima, build/run vLLM CPU, `vllm serve openbmb/MiniCPM5-1B` (fallback model if the architecture is unsupported).
- `tests/integration/test_vllm.py` (env `FOVEATE_TEST_VLLM_URL`, skipped otherwise): real calls through `OpenAIBackend`, and an in-container run of the in-process `VLLMBackend`. Fix `foveate/backends/providers.py` if the real `vllm.LLM.chat` signature differs.

### 8. Documentation rebuild (onboarding-first)
Delete duplicates; create `docs/`: `index.md` (why/who/when/where/how/what), `quickstart.md` (5 min: install, first long-doc question, first benchmark), `concepts.md` (fovea/parafovea/periphery, slots, selection, grounding), `guides/` (long documents and page control, agents and tools, local models/vLLM, evaluating your own pipeline, production), `benchmarks.md` (real numbers), `reference/` (API pages for `documents`, `selection`, `assembly`, `grounding`, `compression`, `backends`, `bench`), `faq.md`, `roadmap.md` (Tier C). README becomes a short pitch + the 200-page example + links. Update `ARCHITECTURE.md`, `PRODUCTION.md` (remove "ceng is already registered on PyPI" text), `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, issue templates, `site/` copy and metrics (only real measured numbers, link to benchmarks), `site/scripts/make-og.py` + regenerate the OG image.

### 9. CI and release readiness
- `.github/workflows/ci.yml`: unit matrix Linux/macOS/Windows x Python 3.10-3.13 (keep), `package` job (wheel contents + clean-venv import), `frameworks` job (installs the three frameworks, runs the no-model adapter tests), `site` job. **Delete the `integration` job and the `NVIDIA_API_KEY` secret logic.** Integration tests stay local, documented in CONTRIBUTING.
- Release workflow publishes `foveate` via trusted publishing (unchanged mechanics, new name).
- Windows-safe: `os.replace`, explicit UTF-8, no symlink assumptions in required tests (already so; verify on the first run).
- Push `v2-rewrite` (force, after history rewrite), open PR, watch all matrix legs, fix failures, then merge to `master` when green.

## Reuse (do not rewrite)
`foveate/compression/*` (strategies, `fit.fair_targets`, `Budget`, `Overflow`), `Runtime.complete` (cache, single-flight, reasoning-model retries, shorten pass), `verification` ABC, `stores`, `okf` persistence, `bench` (`Benchmark`, `Runner`, `Arm`, `BenchResult`), `integrations.history`, `internals.looplocal`, `tests/faults.py` (unit fault injection only).

## Risks / open items
- API cost/latency of full-context baselines (100k+ tokens x many questions) and free-tier rate limits: mitigated by sampling, caching, `--dry-run`, resumable runs; may need a smaller default sample.
- Some filings may exceed the answer model's window: counted as a baseline failure, reported separately from "fits" subset.
- FinanceBench license is non-commercial: benchmark use only, no redistribution of documents.
- MiniCPM5 may be unsupported by the installed vLLM: documented fallback model.
- GitHub push/merge depends on the `sachn-cs` token having write access to `sachncs/context`; if denied I stop and report. Renaming the GitHub repo to `foveate` is recommended but left for the owner.
- Scope is large: phases are ordered so each is shippable and green on its own (0, 1, 2, 3, 6 are the critical path; 4, 5, 7, 8, 9 follow).

## Verification
- `make check`: ruff, `mypy --strict`, pytest (90% coverage gate), no-underscore gate, wheel contains only `foveate/`.
- Unit tests per phase (loaders on generated PDFs, BM25 ranking properties via hypothesis, foveation tier allocation never exceeds budget, citation verifier on crafted answers); fault-injection tests stay in `tests/faults.py`.
- Real-model tests locally: `pytest tests/integration` (long-document pipeline on a real filing; frameworks; vLLM).
- Benchmark: `python -m foveate.bench.longdoc --dry-run`, then the sampled run; report regenerated into `docs/benchmarks.md`.
- Docs: every code block in `README.md`/`docs/` executed by a docs test where it needs no key; links checked.
- CI: PR shows green Linux/macOS/Windows x 3.10-3.13, `package`, `frameworks`, `site`; then merge.
