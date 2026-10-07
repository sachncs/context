ceng v2: Production rewrite around Context(...).compress(method=...) -> Context

Context

ceng (import name; PyPI ceng-context) is ~3k lines of synchronous helper functions. A static review found:

- Correctness bugs. The benchmark "ceng" arm never injects a playbook, so it measures the baseline. Eval fixtures aren't in the wheel. Evolver cache keys collide. Curator bullet IDs collide (cng-00000). budget_tokens is never enforced. Compaction provenance is zeroed. parse_probability("50%") returns 1.0. Integer 0 answers are dropped. Note tags and timestamps aren't persisted.
- Structure. About 60 underscore-prefixed names. Thin wrappers (ppa_compress, parse_playbook, playbooks/, presets.py aliases). Legacy flags (index_only, fallback_to_last). Five copies of cache-then-call. list[dict] messages everywhere. Hidden global backend singleton. Cache connections are never closed. Logging exists in one module only. No concurrency, usage/cost accounting, or budget guarantees.

Goal: a breaking v2.0.0 with one public noun, Context, and a polymorphic strategy hierarchy behind it. No shims, no back-compat.

ctx = Context(messages=[...], runtime=Runtime.from_env())
out = ctx.compress(method="ppa", budget=4000) # -> Context (new, frozen, carries CompressionReport)
out = await ctx.acompress(method="ppa", budget=4000)
out = ctx.compress(method="ushape+ppa", budget=4000) # pipelines
verdict = out.verify(method="macro_fallacy", ...)
out.save("dir", format="okf"); Context.load("dir", format="okf")

Decisions (confirmed with user)

- Scope: re-home everything as Context capabilities or strategies (ppa_check → verify, compaction → a compression strategy, OKF → persistence format, notes → offload store, ACE evolver → ceng.evolution, eval → ceng.bench).
- Async-first: one native async implementation. Sync calls go through a single runner.
- Frozen dataclasses. compress returns a new Context carrying a CompressionReport.

Style rules (Google Python Style Guide, user overrides)

- No leading-underscore names at all. No \_helper, \_CONST, \_singleton. Dunders (**init**, **all**) are allowed.
- Public API is delimited by (a) explicit **all** in every package **init**, (b) an internals subpackage per area for non-API helpers, and (c) docs. Google style normally uses \_ for protected names, so this is a deliberate deviation. Stating it in CONTRIBUTING.md is part of the plan.
- Enforcement: tests/test*style.py AST-scans src/ and fails on any def, class, assignment or attribute starting with a single *. The same script runs in CI.
- Google rules: ruff with pydocstyle convention = "google", 80-column lines, Google-style docstrings (Args/Returns/Raises), absolute imports of modules (from ceng.backends import base), full type annotations (mypy --strict, no Any in public signatures), @dataclass(frozen=True, slots=True) by default, enum.Enum for closed sets, no mutable defaults, no module-level mutable state, no bare except Exception.
- OOP: ABCs for every extension point. Registries are explicit class-level Registry[T] objects (instances, not module globals), populated by a @Compressor.register("ppa")-style decorator.

Target layout (src/ceng/)

context.py Context (frozen), Runtime (frozen: backend, cache, tokenizer, limits, observers)
messages.py Role(Enum), Message (frozen), ContentPart; Message.from_mapping/to_mapping at API edges
errors.py CengError → ConfigError, BackendError(Transient|Permanent|Timeout|RateLimit),
BudgetExceededError, ValidationError, CompressionError(step, cause)
tokenizers/ Tokenizer ABC; HeuristicTokenizer, TiktokenTokenizer; Registry; model→tokenizer resolution
backends/ Backend ABC (async complete(Request)->Completion{text, Usage, model, latency});
LiteLLMBackend, OpenAIBackend, VLLMBackend, ScriptedBackend (deterministic, for tests);
Resilient (retry policy, circuit breaker, concurrency limiter, per-call timeout) composed
as a real decorator (adds behaviour), error classification maps provider exceptions →
Transient/Permanent (auth/bad-request never retried)
cache/ Cache ABC (async get/set, typed CacheKey/CacheEntry); SqliteCache (WAL, corruption-tolerant,
TTL + max-bytes eviction, close/context manager), MemoryCache, NullCache
CacheKey = sha256 over full request fingerprint: model, rendered messages, system prompt,
sampling params, PromptTemplate.fingerprint, strategy name+version
prompts.py PromptTemplate (frozen: name, version, system, user_fmt, delimiters; fingerprint property)
partition/ Partitioner ABC; RecursivePartitioner (sentence→paragraph→word, preserves separators/newlines),
FixedWindowPartitioner; Partition (frozen: text, index, span, tokens)
compression/ Compressor ABC (see below), Registry, CompressionReport, StepRecord, Budget
ppa.py PartitionSummarizeCombine (concurrent leaves, semaphore-bounded)
hierarchical.py Recursive PPA until Budget satisfied (fixes unenforced budget)
ushape.py Keep head/tail, compress/drop middle (old compact_messages)
window.py SlidingWindow; truncate.py Truncate(head|tail|middle)
extractive.py LLM-free sentence ranking (cheap fallback, offline)
offload.py Writes dropped content to a NotesStore, leaves retrieval pointer
pipeline.py Pipeline([Compressor,...]); "a+b" method strings parse into it
fallback.py FallbackCompressor(primary, secondary) for degraded mode on BackendError
stores/ NotesStore ABC; FilesystemNotesStore (atomic os.replace, file lock, persisted tags/timestamps)
verification/ Verifier ABC; MacroFallacyVerifier (old ppa_check), TreeNode public, Verdict, LeafEstimate;
validates root sums; fixes percent parsing
okf/ Concept, Frontmatter, Codec ABC; OkfCodec (read/write bundle, staged atomic swap, symlink guard);
Context.save/load dispatch on format registry
evolution/ Playbook, Bullet (UUID ids, no collision), CuratorOp ABC → AddOp/UpdateOp/MergeOp/DeleteOp,
Generator/Reflector/Curator roles as classes, Evolver (checkpoint/resume, curator_frequency
honored, correctness via Benchmark.is_correct, helpful/harmful on success+failure)
bench/ Benchmark ABC (load_samples, build_prompt, is_correct, seed_playbook); Finer/Formula/DDXPlus/
AppWorld subclasses (AppWorld either implemented or removed; no NotImplementedError stubs);
Runner (concurrent, writes json/md); fixtures shipped via package-data; arm "ceng" actually
injects the (evolved) playbook
observability.py Observer ABC (on_step_start/end, on_cache, on_retry); LoggingObserver, MetricsObserver
(usage/cost/latency aggregates); module loggers with NullHandler only
cli.py `ceng compress|verify|convert|bench` (argparse, JSON in/out, exit codes); entry point `ceng`

Core abstractions

@dataclasses.dataclass(frozen=True, slots=True)
class Context:
messages: tuple[Message, ...]
runtime: Runtime = dataclasses.field(default_factory=Runtime.from_env)
report: CompressionReport | None = None
metadata: Mapping[str, str] = MappingProxyType({})
def compress(self, method: str | Compressor = "ppa", \*, budget: int | Budget, **options) -> Context
async def acompress(...) -> Context
def verify(self, method="macro_fallacy", **options) -> Verdict
def save(self, path, format="okf") -> None ; @classmethod load(...)
@property token_count

class Compressor(abc.ABC): # template method: validate → short-circuit if within budget → run → enforce budget → report
name: ClassVar[str]; version: ClassVar[str]
@abc.abstractmethod async def run(self, context, budget) -> Context

- method string resolves via Compressor.registry. Per-strategy options are typed CompressorConfig dataclasses (validated at construction), accepted as kwargs or as an instance. Unknown option → ConfigError.
- Budget enforcement is in the base class: after run, if tokens > budget it invokes the strategy's tighten() hook (hierarchical re-pass), then falls back to the overflow policy (raise | truncate), so the guarantee is explicit.
- Runtime replaces the global backend singleton and every cache_dir/llm string argument. There is no hidden state, so tests inject ScriptedBackend + MemoryCache.
- The single sync runner is ceng.internals.runner.run_sync(coro). It uses asyncio.run normally and a worker thread when a loop is already running. This is the only sync/async bridge.

Resilience and reliability (cross-cutting)

- Retry with exponential backoff and full jitter, only for TransientBackendError, honoring Retry-After. Per-request timeout and overall deadline. Circuit breaker. Bounded concurrency (semaphore) and optional rate limit.
- Cache stampede protection (single-flight per key). Cache corruption is treated as a miss and logged.
- Partial progress: leaf results are cached as they complete, so a retry resumes. FallbackCompressor degrades to extractive when the backend is down (opt-in).
- Output validation: empty, None or over-length completions raise ValidationError, with a bounded regeneration attempt.
- Resource hygiene: all caches and stores are context managers, and Runtime.aclose() releases them.
- Usage and cost: Usage(prompt, completion) from the provider response (heuristic fallback), aggregated in CompressionReport with a pluggable price table.
- Config comes from Runtime.from_env() with strict parsing (bad env values raise ConfigError, not silently ignored).

Phases (each lands green: ruff, mypy --strict, pytest ≥ 90% branch coverage)

1. Foundation: tooling and style gate (ruff google, mypy strict, test_style.py no-underscore scan), errors, messages, tokenizers, observability, prompts. Delete the old modules as their replacements land. There is no parallel old/new code.
2. I/O layer: backends (+ resilience, error classification), cache, Runtime, ScriptedBackend. Tests for retry/timeout/breaker/corruption/single-flight.
3. Core: partition, Compressor ABC + registry + budget enforcement + report, ppa, hierarchical, ushape, window, truncate, extractive, pipeline, fallback, then Context with compress/acompress.
4. Persistence and tools: okf codec + Context.save/load, stores + offload, verification + Context.verify.
5. Evolution and bench: evolution (ID, cache-key, gating and frequency fixes), bench with real playbook injection and shipped fixtures, cli.
6. Release: rewrite README/ARCHITECTURE/PRODUCTION/CHANGELOG (2.0.0, migration notes only as a "what was removed" list), fix CI (macOS+Windows matrix, pinned actions, release gated on test job, pre-commit config, drop unused pytest-asyncio, honest classifier "Beta", bounded dependency ranges), update examples/ and the site's code snippets.

Old → new mapping: ppa_compress/compress_to_bundle → Context.compress("ppa"); ppa_compress_to_okf → compress(...).save(format="okf"); compact_messages → method="ushape"; ppa_check → Context.verify; NotesManager → FilesystemNotesStore; Playbook\*/Evolver → ceng.evolution; ceng-bench → ceng bench. The presets.py aliases and playbooks/ are deleted.

Critical files

Replaced/deleted: all of src/ceng/_.py, compress/, playbook/, playbooks/, eval/, presets.py, bench.py. Reused with rewrite (logic worth keeping): okf.py (parse/render and the staged-swap writer), partition.py (recursion idea), backends.retry_with_backoff (jitter logic), cache.py (sqlite WAL + schema versioning), playbook/prompts.py and compress/prompts.py (prompt text). Also touched: pyproject.toml (package-data for fixtures, extras, entry point ceng, tool config), .github/workflows/{ci,release}.yml, Makefile, docs, examples/_.

Verification

- make check: ruff check, ruff format --check, mypy --strict src/ceng, pytest --cov --cov-branch --cov-fail-under=90. test_style.py must report zero underscore-prefixed names.
- Regression tests, one per bug in the review: budget enforced (output ≤ budget or BudgetExceededError), cache-key sensitivity (changing the prompt/system/temperature/playbook changes the key), unique bullet IDs, curator_frequency honored, 50% → 0.5, answer 0 kept, notes tags round-trip, compaction report token counts non-zero, newline preservation in the partitioner, fixtures present in the built wheel.
- Resilience tests with ScriptedBackend fault injection: transient-then-success, permanent failure (not retried), timeout, breaker open/half-open, concurrent identical requests hit the backend once, corrupted cache row.
- Property tests (hypothesis): partition concatenation preserves text; partitions ≤ max tokens; pipeline output ≤ budget.
- Build wheel, install into a clean venv outside the checkout, run ceng compress on a sample JSON and the offline examples/; confirm ceng bench smoke shows a nonzero delta only when a playbook is injected.
- CI green on Linux/macOS/Windows × Python 3.10–3.13.
