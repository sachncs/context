# Foveate round 2: research-backed scope, needle benchmarks, compression proof, launch-ready site

## Context

Round 1 (rename, long-document pipeline, agent tools, benchmark harness, vLLM, docs rebuild, CI without secrets) is
committed on `v2-rewrite` (8 commits, 462 tests, 95% coverage). Still open from round 1: the real benchmark run
(`scripts/run_longdoc.py --n 30 --runs 3`, running in the background, writes `bench/results/run1/`), `docs/benchmarks.md`,
README results block, push, PR, CI on three OSes, merge.

The owner's new review: context engineering is bigger than selection plus citations; provide research grounding
(arXiv 2025-2026, today is Oct 2026); add needle-in-a-haystack (NIAH) evaluation; make **compression** the headline
proof; delete the old site `https://sachncs.github.io/context/` and its Pages deployment; the site is badly composed,
the copy is weak, the docs are thin, UX/UI is not launch-ready. Decisions (confirmed): site on **GitHub Pages of the
renamed repo** (owner renames the repo to `foveate` last), docs via **Astro Starlight from `docs/`**, NIAH set =
**NoLiMa + our RULER-style generator + a LongBench v2 sample**.

## Research map (what "context engineering" covers, 2025-2026)

Verified by search; used for the docs page `docs/research.md` (one paragraph each, what it implies for Foveate) and to
drive the gap list below. IDs are arXiv unless noted.

| Area | Papers | Takeaway for Foveate |
|---|---|---|
| Definitions, surveys | Context Engineering 2.0 (2510.26493); Survey of Context Engineering for LLMs (2507.13334); Externalization in LLM Agents (2604.08224) | Context = selection, structure, compression, memory, tools, isolation. Position Foveate as that whole pipeline, not only compression. |
| Context rot, position effects | Diagnosing and Mitigating Context Rot in Long-horizon Search (2606.29718); Classifier Context Rot (2605.12366); Intelligence Degradation in Long-Context LLMs (2601.15300); Lost in the Middle (2023, classic) | Longer is worse past a threshold; models give up early. Motivates budgets, position-aware ordering, abstention. |
| NIAH and long-context evals | NoLiMa (2502.05167, non-literal needles: 10/12 models fall below 50% of baseline at 32K); BABILong (2406.10149); Sequential-NIAH (2504.04713); RULER and LongBench v2 (earlier, still the standard) | Literal needles are too easy; add non-literal needles, multi-needle, length x depth grids. |
| Prompt and context compression | End-to-End Context Compression at Scale (2606.09659); AdmTree (2512.04550); Evaluator Heads (2501.12959); KV-cache compression benchmarks (2407.01527, 2510.00636, 2510.13334) | Compression ratio vs accuracy is the metric; most work is model-internal (KV), Foveate is the model-agnostic text layer. |
| Query-aware pruning for RAG | Provence (ICLR 2025); AttentionRAG (2503.10720); EnComp/LooComp (2603.09222) | Prune per sentence by query relevance, not by frequency: upgrade the CONDENSED tier. |
| Agent context management | ACON (2510.00615, 26-54% fewer peak tokens, >95% accuracy kept); Less Context, Better Agents (2606.10209, 8% to 91.6% task completion with pruning plus summaries); ACM (2607.23809); TokenPilot cache-efficient context (2606.17016) | History/tool-output compression is validated; keep prefixes stable so provider prompt caches hit. |
| Self-improving contexts | ACE (2510.04618, ICLR 2026); Meta Context Engineering (2601.21557) | Already in `foveate.evolution`; document and benchmark it. |
| Agent memory | A-MEM (2502.12110); Anatomy of Agentic Memory (2602.19320); Survey of Agent Memory (2602.06052); Agentic Memory (2601.01885) | Long-term memory with recall is a first-class part of context engineering (was Tier B). |
| Structure and format | Structured Context Engineering for File-Native Agents (2602.05447) | Format choices change accuracy; document recommended page/outline formats. |

More papers found in the second pass: LOCA-bench, controllable extreme context growth for agents (2602.07962);
Don't Break the Cache, an evaluation of prompt caching (2601.06007); Harness Engineering for Agentic Coding Tools
(2602.14690); Building Effective Terminal Coding Agents (2603.05344); AI Agents Do Not Fail Alone: The Context Fails
First (2607.14275); SmoothAgent lookahead context engineering (2607.00151).

I will re-verify each arXiv id and one-line claim when writing the page and drop anything I cannot confirm.

### References supplied by the owner (all go on `docs/research.md` and the site, verbatim citations)
Wolf et al. 2026, Partition, Prompt, Aggregate (arXiv:2607.15277, basis of `ppa`); Zhang et al. 2026, ACE (arXiv:2510.04618,
code github.com/ace-agent/ace); Hua et al. 2025, Context Engineering 2.0 (arXiv:2510.26493); Anthropic 2025, Effective
context engineering for AI agents; McVeety and Hormati 2026, Open Knowledge Format (Google Cloud Blog, basis of `okf`);
Coyle 2025, Context Engineering: The New AI Frontier (Medium); Szapar 2026, Context Engineering Research: 2026 Papers and
Benchmarks (iwoszapar.com). Grey literature (the last two and any blog) is labelled as such and not used for numeric claims.

### Industry playbooks and rulebooks (read, distil into `docs/playbook.md`, cite every rule)
| Source | What it contributes | Maps to |
|---|---|---|
| **Anthropic**: Effective context engineering for AI agents; context-management tools (compaction, tool-result clearing, memory tool, context awareness, programmatic tool calling); just-in-time retrieval | Compact when near the limit, clear old re-fetchable tool results but keep the record of the call, structured note-taking memory, let the agent pull context on demand, tell the model how much room is left | `ppa`/`ushape` compaction, G7 `clear_tool_results`, G4 memory, `read_pages`/`search_document` tools, G8 context awareness |
| **OpenAI**: A practical guide to building agents (2025); Agents SDK session memory cookbook (trim last N turns vs summarise); prompt caching guidance | Start with one agent, instructions from existing docs, layered guardrails; deterministic trimming is easier to reproduce than summaries; cache by stable prefixes | `window` and `ushape` defaults, G3 cache-stable prompts, injection fencing, docs guide "single agent first" |
| **Google**: Context Engineering: Sessions and Memory whitepaper (Nov 2025, Kaggle agents course day 3); Google Cloud agentic architecture guide; ADK sessions, state, memory services | Sessions (working memory) vs Memory (consolidated, cross-session); assemble context dynamically; memory extraction and consolidation | G4 `Memory` (session vs long-term split), ADK adapter, `Assembly` slots |
| **Amazon/AWS**: Bedrock AgentCore Memory and AWS Prescriptive Guidance on agentic frameworks; Strands Agents `SlidingWindow` and `Summarizing` conversation managers | Production memory patterns, window plus summarise as the default overflow policy | `ushape|extractive` default, `Fallback` composition, G4 |
| **Uber**: Genie enhanced agentic RAG (on-call copilot; reported +27% acceptable answers, -60% incorrect advice vs plain RAG); Prompt Engineering Toolkit; Finch; evaluation-first culture for the agent platform (2026) | Rich document preprocessing, pre- and post-processing agents (query rewrite, source identification, context refinement), golden test sets from experts, evals on day one | `Loader` structure/headings, G2 query-aware pruning, `rerank`, benchmark guide "start with 100 golden questions", `foveate.bench.longdoc` as the eval-first harness |
| **Manus** (public engineering write-up, 2025) | Keep prompts append-only and deterministic for KV-cache hits, mask tools rather than remove them, use the file system as unlimited context, keep errors in context, recite goals | G3, G7, `offload` (file-system style), docs rules |

Only claims I can confirm from the primary page go in; numbers above marked "reported" are re-checked against the source
before publication, otherwise dropped. Pages to fetch and verify: each source's primary URL (Anthropic engineering post,
OpenAI PDF, Google/Kaggle whitepaper PDF, AWS Prescriptive Guidance page, Uber engineering blog posts, Manus blog).

### The rulebook (`docs/playbook.md`, new)
"Twelve rules of context engineering", each: the rule, the source(s), why it works, the Foveate feature or setting that
implements it, and a failure it prevents. Draft list: (1) measure first, with a golden set; (2) send less, not more;
(3) put the answer-bearing text at the edges; (4) keep prefixes stable and append-only; (5) trim deterministically before you
summarise; (6) clear re-fetchable tool output, keep the record; (7) pull context just in time with page tools; (8) separate
working memory from long-term memory; (9) cite and verify; (10) abstain when the evidence is missing; (11) fence retrieved
text as data; (12) tell the model how much room it has. Each rule links to the benchmark that shows its effect where we have one.

## Gaps this exposes -> what to build

| # | Feature | Why (paper) | Where |
|---|---|---|---|
| G1 | **Position-aware assembly**: put the strongest pages first and last, weakest in the middle (option `order="edges"`, default for FULL pages) | Lost in the middle, context rot | `foveate/foveation.py`, `foveate/foveator.py` render order |
| G2 | **Query-aware CONDENSED tier**: score sentences by BM25/embedding similarity to the question instead of word frequency | Provence, AttentionRAG, EnComp | new `compression/queryaware.py` (`Compressor`, name `query`), used by `foveation.allocate` |
| G3 | **Cache-stable prompts**: static instructions and outline first, volatile pages last, no timestamps in prefixes; expose `Answer.usage.cached_tokens` when the provider reports it | TokenPilot | `foveate/foveator.py`, `foveate/usage.py` |
| G4 | **Long-term memory**: `Memory` on top of `stores.NotesStore` (add, summarise, recall by query, evict), usable as an assembly slot | A-MEM, Agentic Memory | new `foveate/memory.py`, wire into `assembly.Slot("memory")` |
| G5 | **Compression guideline optimisation** (ACON style) using the existing ACE loop on compression prompts | ACON | stretch; only if time remains after the benchmark |
| G6 | **Sufficiency signal**: `Answer.support` = share of cited quotes verified, plus `needs_more_context` when the model asks for pages and none remain | Provence/CRAG line of work | `grounding/answer.py` |

| G7 | **Tool-result clearing**: replace old, re-fetchable tool results with a stub that keeps the call and its arguments (`clear_tool_results`, keep last N) | Anthropic context management, Manus | new `compression/clearing.py`, registered as a `Compressor`, composable (`clear_tool_results+ushape`) |
| G8 | **Context awareness**: optional one-line note to the model with tokens used and remaining (`Budget` slot), refreshed per call | Anthropic context awareness | `assembly.py`, `integrations/history.py` option |
| G9 | **Session vs long-term memory split** (working notes per session, consolidated facts across sessions, extraction + consolidation step) | Google sessions/memory whitepaper, AgentCore | part of G4 `Memory` design |
| G10 | **Eval-first starter**: `foveate.bench.longdoc` gains `quick_eval(documents, questions)` that builds a golden set skeleton (answerable plus synthetic unanswerable and needle items) from the user's own files | Uber evaluation-first culture | `bench/longdoc/gold.py`, guide `evaluating.md` |

G1-G3, G6-G8 are small and in scope; G4/G9 and G10 are in scope; G5 is stretch.

## Benchmarks

### A. Needle-in-a-haystack (new, `foveate/bench/needle/`)
- **Datasets**: (1) **NoLiMa** public data (HF `amodaresi/NoLiMa`; check license before use, download at run time, never redistribute); (2) our **RULER-style generator** (seeded, any length): single needle, multi-needle (k=2,4), multi-key distractors, variable tracking, using FinanceBench filings as realistic haystacks and a neutral prose haystack; (3) **LongBench v2** sample of 30 (HF `THUDM/LongBench-v2`, check license), multiple choice.
- **Grid**: context length {8k, 16k, 32k, 64k, 128k} x depth {0, 25, 50, 75, 100%} x pipeline {full-context, truncate, naive-rag, foveate}; a budget of 4k for the limited pipelines; 3 seeds on the small cells, 1 seed on the large.
- **Outputs**: heatmaps (length x depth accuracy per pipeline), accuracy vs length line chart, tokens and latency, JSON in `bench/results/needle/`.
- Reuses: `bench/longdoc/gold.needle_items/with_needle`, `pipelines.Pipeline` registry, `metrics.numeric_match`, `runner.Report`.

### B. Compression showcase (the headline, `foveate/bench/compression/`)
- **Question**: how much can the context shrink before answers degrade? Sweep compression ratio {1x, 2x, 4x, 8x, 16x} on (i) long-document QA (FinanceBench sample, answer-bearing pages kept or not) and (ii) NIAH cells above (iii) agent history (multi-turn chat with a planted fact) and tool output (large JSON with a planted field).
- **Methods compared** (same budget, same answer prompt): `truncate`, `window`, `extractive`, `ushape`, `ppa`, `query` (G2), `tool_output`, the full `foveate` pipeline, and optionally **LLMLingua-2** as an external baseline behind an extra (`foveate[bench]`; skip and say so if it will not install).
- **Metrics**: answer accuracy, planted-fact retention, tokens saved, compression ratio actually achieved, extra model calls and cost for the compressor, latency. Chart: accuracy vs compression ratio per method (the "Pareto" figure for the site and README).
- Honesty gate as in round 1: publish results including losses; if `foveate` loses to a baseline at some ratio, say so and fix the method before publishing.

### C. Finish round 1 runs
`bench/results/run1` -> `docs/benchmarks.md`, README block; small-model (MiniCPM5-1B via vLLM, 8k window) run of the same long-document set on 10 questions where full-context cannot run (CPU is slow, ~12 tok/s: small n, stated).

## Website (delete old, rebuild)

1. **Delete the old deployment** (outward-facing, explicitly requested): `gh api -X DELETE repos/sachncs/context/pages` (current Pages is `legacy`, source `master:/`); remove `.github/workflows/pages.yml` deploy jobs. Keep a `site` CI job that builds the site (no deploy) until the repo is renamed; add a `pages.yml` that deploys on `push` to `master` **only after the owner renames and enables Pages (source: GitHub Actions)**. I will say clearly in the final message that the new URL appears after the rename. Do this deletion as the last step before merge so the old URL does not 404 while the PR is open? No: the owner asked for it deleted, so delete when the PR merges; I will confirm by running the command only at that point.
2. **Stack**: Astro 7 + **Starlight** for `/docs`, custom landing page, Tailwind already present. `base` is `/foveate/`, `site` `https://sachncs.github.io`; `astro.config.mjs` reads both from env so a custom domain is a one-line change.
3. **Single source for docs**: Starlight content collection with a glob loader pointing at `../docs` (no copies, no symlinks, Windows-safe); add frontmatter (`title`, `description`) to every `docs/**/*.md`; `tests/test_docs.py` keeps link and example checks and gains a frontmatter check.
4. **Information architecture** (sidebar): Start (Overview, Quickstart, Install) / Concepts (Pages and foveation, Selection, Grounding, Compression, Memory, Agents, Reliability) / Guides (existing five plus Compress chat history, Compress tool output, Run the benchmark on your documents) / Evidence (Benchmarks: long documents, Needle-in-a-haystack, Compression; Research map) / Reference (API by module, Environment variables, Errors) / Project (FAQ, Roadmap, Changelog, Contributing). Every concept page: the problem, a picture, the minimal code, the knobs, the failure modes.
5. **Landing page** (rewrite, not patch): hero with one sentence of value and the 6-line example; "what happens to your 200 pages" strip showing real numbers from the benchmark JSON; three interactive explainers; results section; agent integrations; install; footer. No stock claims; every number is read from `bench/results/**.json` at build time by `site/scripts/sync-results.mjs`, so the site cannot drift from the benchmarks.
6. **Interactive explorers** (small Astro islands, no framework runtime unless needed):
   - *Foveation explorer*: a 200-page strip, a question picker (3 canned questions with real precomputed `Plan` output) and a budget slider; pages colour by tier (FULL/CONDENSED/OUTLINE/DROPPED); shows tokens sent vs document tokens.
   - *Compression explorer*: method and ratio picker showing before/after text from real runs and the accuracy-vs-ratio chart.
   - *Needle heatmap*: pipeline tabs, length x depth grid with accuracy.
   - Precomputed fixtures in `site/src/data/` generated by `site/scripts/make-fixtures.py` from the library (no model needed for the plan fixtures).
7. **Design and UX**: one coherent system (type scale, 8px spacing grid, a restrained palette with an accent, light and dark themes, visible focus states, `prefers-reduced-motion`), mobile-first with a 16px gutter, no horizontal scroll, semantic landmarks, skip link, alt text, contrast >= 4.5:1, code blocks with copy buttons, per-page "edit on GitHub" and prev/next, search (Starlight Pagefind), `404` page, sitemap, OG image per section, favicon set. Remove the current decorative glow/grid blocks and dead components.
8. **Voice**: plain, specific, active; short sentences; say what it does, then what it costs, then where it fails; no superlatives or marketing filler; consistent terms (page, tier, budget, citation, grounded, abstain); one-sentence summary at the top of every page; every claim links to a benchmark or paper. I will do a copy pass over README, docs, site and error messages against this checklist and keep a `docs/style.md` for contributors.
9. **Delete** the old site components that are unused after the rebuild, `site/public/og.png` regenerated by `make-og.py`.

## Phases and order (site first, as the owner asked)

**Housekeeping at the start**: replace the repo-root `plan.md` with this plan (it is the goal file; the old text is obsolete).

S. **Build the site first** (Website section below): Starlight docs from `docs/`, redesigned landing page, explorers, and a
   **Coming next** page. Real benchmark numbers are not final yet, so the first launch-ready version reads what exists
   (`bench/results/run1` once finished, the 6-question smoke run is NOT used) and every number block degrades to
   "results pending" with the method described, never invented figures; `sync-results.mjs` fills them as each benchmark lands.
   - **Coming next** (`/roadmap/`, linked from the nav and the landing page): a visual roadmap with status chips
     (Shipping now / In progress / Planned / Exploring) for position-aware assembly, query-aware pruning, cache-stable prompts,
     tool-result clearing, long-term memory, needle and compression benchmarks, ACON-style guideline optimisation, table-aware
     PDFs, streaming, vector-DB connectors, multimodal pages and OCR, PII redaction, TypeScript SDK, hosted eval dashboard.
     Each item: what it is, why (paper or playbook link), how you will notice it, and a "vote on GitHub" link. Source of truth is
     `docs/roadmap.md` (frontmatter list rendered into cards) so the repo and the site never disagree.
   - Site acceptance: a first-time visitor can answer in under a minute what Foveate does, whether it fits them, how to install,
     and what is coming; all pages pass the QA checklist under Verification.
0. Finish round 1: wait for `run1`, read results, write `docs/benchmarks.md` and the README block, commit. (Iterate the pipeline first if foveate loses to naive RAG on a target metric.) The site then picks up the numbers.
1. G1, G2, G3, G6 with unit tests (offline, deterministic); G4 memory with tests; re-run the small benchmark to see their effect (publish deltas).
2. Needle benchmark (A) and compression benchmark (B): loaders, generators, runners, reports, unit tests with the fault backend, then real runs on `gpt-oss-20b` (resumable, cached, `--dry-run` estimate first; sample sizes stated). Optional small-model needle run on vLLM at 8k.
3. Fetch and verify every primary source (arXiv abstracts, Anthropic/OpenAI/Google/AWS/Uber/Manus pages); write
   `docs/research.md` (paper map plus the seven owner references plus industry sources, each with a link and a one-line
   "what Foveate takes from it") and `docs/playbook.md` (the twelve rules); new concept/guide pages; copy pass. The site
   gets both under Evidence and links each rule to its benchmark. Add G7, G8, G10 with tests.
4. Site refresh with the new evidence: research, playbook and benchmark pages, needle heatmap and compression chart wired to
   the final JSON; re-run visual QA; mark shipped roadmap items as Shipping now.
5. Git (the site work in step S is committed and pushed early on its own commits so CI proves the build; the final merge waits for
   the benchmarks): rebase on current branch, no trailers, force-push `v2-rewrite`, open PR (no generated-by footer), watch Linux/macOS/Windows x 3.10-3.13 plus `package`, `frameworks`, `site`; fix; merge when green; then delete the old Pages deployment and tell the owner to rename the repo to `foveate` and enable Pages (Actions source). Recommend (not do) the rename.
6. Update the memory file with the Foveate status (never the API key).

## Critical files

`foveate/foveation.py`, `foveate/foveator.py`, `foveate/compression/queryaware.py` (new), `foveate/memory.py` (new),
`foveate/grounding/answer.py`, `foveate/usage.py`, `foveate/bench/needle/` (new), `foveate/bench/compression/` (new),
`foveate/bench/longdoc/*`, `scripts/run_needle.py`, `scripts/run_compression.py`, `docs/**`, `site/**`,
`.github/workflows/{ci,pages}.yml`, `pyproject.toml` (extras `bench`), `tests/**`.

## Reuse (do not rewrite)
`Foveator`, `Plan`, `allocate`, `Runtime.complete`, `Context.compress` strategies, `compression.fit`, `bench.longdoc`
(`gold`, `pipelines`, `metrics`, `runner`, `dataset`), `stores.NotesStore`, `evolution` (ACE), `integrations.tools`,
`tracing.TracingObserver`, `tests/faults.py`, `scripts/build_gold.py`.

## Risks
- NoLiMa/LongBench v2 licenses or download changes: check each license first, download at run time, skip with a clear message if unavailable; the generator keeps the NIAH set self-contained.
- 128k-token cells on a free-tier API are slow and costly: sample, cache, resume, cap concurrency, state n; 128k cells run with 1 seed.
- LLMLingua-2 may not install on Python 3.14/macOS: optional, documented, skipped otherwise.
- Starlight with a docs folder outside `site/` needs a custom loader base; fall back to a build-time copy script if the glob loader rejects parent paths.
- Pages URL will not exist until the owner renames the repo and enables Pages; the site CI job proves it builds.
- Deleting the old Pages deployment is irreversible for that URL; done last, only after the PR merges.
- Small CPU models are slow (about 12 tok/s): small n, stated plainly.

## Verification
- `make check` (ruff, `mypy --strict`, pytest, 90% coverage, no-underscore gate) after every phase; `pytest tests/test_docs.py` (links, runnable examples, frontmatter, stale names).
- Unit tests: position ordering property (best page at an edge), query-aware pruning keeps the sentence containing the answer, cache-stable prefix identical across questions, memory add/recall/evict, needle generator determinism and depth placement, compression runner report math.
- Real runs: needle grid and compression sweep to `bench/results/`; numbers on the site are generated from those JSON files (`sync-results.mjs`), build fails if a referenced file is missing.
- Site: `npm run build`; Playwright screenshots at 390, 768, 1280 px in light and dark; keyboard-only walkthrough; Lighthouse (performance, accessibility, SEO >= 95) on the landing page and one docs page; link checker over the built output.
- CI: PR green on Linux/macOS/Windows x 3.10-3.13, `package`, `frameworks`, `site`; merge; then run the Pages deletion and report the manual rename steps.

## Round 2b: nine more papers, read and applied (owner request, 8 October 2026)

Read in full or in part and checked on 8 October 2026: IC-Former (arXiv:2406.13618), Beyond Token
Savings (2609.32961), FOCUS (2609.37590), End-to-End Context Compression at Scale / LCLM (2606.09659),
Thinking as Compression (2605.28713), Compress the Context, Keep the Commitments (2605.17304),
Semantic-Anchor Compression (2510.08907), ComprExIT (2602.03784), Selective Context (2310.06201).

| Paper | What it teaches | What Foveate does with it |
|---|---|---|
| Beyond Token Savings | Policy = primitive x trigger x depth; token savings do not mean lower latency or billed cost (prefix caching); rankings flip across models; same aggregate score hides task turnover; partial structured rewriting that keeps recent history verbatim is consistently good | Compression benchmark reports compressor calls, latency, billed cost with cached-token price, and per-sample gained/lost turnover; `HistoryCompressor` gets `trigger` and `target` knobs; `PriceTable` gets a cached-token price |
| FOCUS | Compress by decision preservation, not redundancy; span-level units; defensive verification rescues negative constraints and state | `clear_tool_results` keeps error results and honours `exclude_tools` |
| Compress the Context, Keep the Commitments | Typed atoms; Critical Atom Recall, Weighted Atom Recall, commitment density; error taxonomy (omission, weakening, mutation, polarity flip, ...); round-trip verification; keep raw spans for low-confidence atoms | New deterministic metrics `critical_atom_recall`, `weighted_atom_recall`, `commitment_density` and a failure taxonomy (omitted / mutated) in the compression benchmark, plus a multi-atom history task |
| Thinking as Compression | Query-conditioned compression; evaluated with EM and F1 at 4x and 8x on NQ, 2WikiMQA, HotpotQA, MuSiQue | Exact match and token F1 metrics; HotpotQA (public, distractor setting) QA task at 1x, 4x, 8x with the `query` compressor |
| Selective Context | Self-information pruning; phrase-level units beat token or sentence; percentile thresholds; 50% compression = -32% latency, -36% memory; evaluate against the full-context answer as reference | New LLM-free `selective` compressor (information per phrase, percentile threshold); metric `agreement_with_full` (token F1 against the full-context answer) |
| LCLM (end-to-end compression at scale) | Learned latent compression; reports time-to-first-token and memory, not just accuracy; agent scaffolding with selective expansion helps needle tasks | Foveate's outline plus `need_pages` is the same selective-expansion idea; needle report includes extra rounds and latency; positioning below |
| IC-Former, SAC, ComprExIT | White-box learned compressors (need hidden states or training); evaluated on MRQA in and out of domain with EM and F1; uniform coverage and complementary (non-redundant) allocation matter | Positioning: Foveate is the black-box layer for API models; coverage-aware selection recorded on the roadmap; EM/F1 protocol adopted |

### Actions
1. Metrics and tasks (offline, tested): EM, token F1, atom recall family, failure taxonomy, `agreement_with_full`, billed cost, turnover; `atoms` and `qa` (HotpotQA) tasks in `foveate.bench.compression`.
2. Product: `selective` compressor; defensive `clear_tool_results`; `trigger`/`target` on `HistoryCompressor`; cached-token price.
3. Experiments (real): compression sweep with the new tasks and metrics; needle grids (literal, NoLiMa-style with embeddings, multi); long-document run 2 with embeddings and expansion against plain RAG using the same retrieval; **small model against large model**: MiniCPM5-1B behind Foveate against a 30B-class or larger model sent the full filing or plain RAG, same questions, same judge; published whichever way it goes.
4. Positioning: `docs/why.md` ("Why context engineering"), a map of where Foveate sits against learned compressors, prompt-level compressors, KV-cache methods and agent context managers, with only measured claims; landing-page section; research page extended with these nine papers.
5. Honest limits: claims such as "a 1B model can beat a 30B model" are tested, not asserted; they appear on the site only with the numbers and sample sizes that support them.
