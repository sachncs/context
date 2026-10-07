# ceng

Context engineering for LLMs, with one noun: **`Context`**.

```python
from ceng import Context, Runtime

context = Context.from_dicts(messages, Runtime.from_env())
smaller = context.compress("ppa", budget=4000)     # -> a new Context

print(smaller.report)                              # tokens, steps, usage, cost
```

`compress` never mutates. It returns a new frozen `Context` that carries a
`CompressionReport`, and it **guarantees the result fits the budget**: either
the context fits, or `BudgetExceededError` is raised, or (opt-in) the result is
hard-truncated and the report says so.

- Package on PyPI: `ceng-context` · import name: `ceng` · Python 3.10-3.13
- v2 is a breaking rewrite; see [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install ceng-context                 # core (PyYAML only)
pip install "ceng-context[litellm]"      # any provider via LiteLLM (default backend)
pip install "ceng-context[openai]"       # OpenAI SDK / compatible servers
pip install "ceng-context[vllm]"         # in-process vLLM
pip install "ceng-context[tokenize]"     # exact token counts with tiktoken
```

Credentials come from the provider's usual environment variables.

## Compression methods

`method` is a registered name, a composition, or a configured instance.

| Method | Needs LLM | What it does |
|---|---|---|
| `ppa` | yes | Partition, summarise leaves concurrently, aggregate (Wolf et al., 2026). |
| `hierarchical` | yes | Repeats PPA rounds until the budget is met. |
| `ushape` | yes* | Keeps head and tail turns; summarises (or drops) the middle. |
| `window` | no | Drops the oldest turns; system messages are kept. |
| `truncate` | no | Cuts oversize messages (`keep=head\|tail\|middle`). |
| `extractive` | no | Keeps the highest-scoring sentences, in order. |
| `offload` | no | Moves old turns to a notes store and leaves a retrieval pointer. |

\* `ushape` with `middle="drop"` needs no LLM.

Budget is shared across messages by max-min fairness: system messages are
protected, messages below the fair share are untouched, larger ones are
compressed to it.

```python
context.compress("ushape+ppa", budget=4000)              # pipeline: stop as soon as it fits
context.compress("ppa|extractive", budget=4000)          # fallback if the LLM path fails
context.compress("ppa", budget=4000, leaf_tokens=512)    # strategy options as kwargs
context.compress("ushape+ppa", budget=4000, ppa={"leaf_tokens": 512})  # per-stage options
context.compress(                                         # hard-truncate instead of raising
    "ppa", budget=Budget(4000, Overflow.TRUNCATE)
)
await context.acompress("ppa", budget=4000)               # native async
```

Add a method by subclassing `Compressor` and decorating it:

```python
@Compressor.register("mine")
@dataclasses.dataclass(frozen=True)
class Mine(Compressor):
    async def run(self, context, budget, trace): ...
```

## Compress without a model

`Runtime.without_llm()` builds a runtime whose backend refuses every call, so
`window`, `truncate`, `extractive` and `offload` (and pipelines of them) run
with no provider, no key and no network:

```python
ctx = Context.from_dicts(messages, Runtime.without_llm())
ctx.compress("window+extractive", budget=4000)
```

## Agent frameworks

`ceng.integrations` puts compression in the history path of three frameworks.
Each keeps the newest turns verbatim, summarises the older ones as one
transcript (falling back to an offline method if the model is unreachable),
and never splits a tool call from its result. Runnable demos are in
[`examples/`](examples/).

| Framework | Hook | Demo |
|---|---|---|
| Pydantic AI (2.x) | `Agent(capabilities=[ProcessHistory(ceng_pai.history_processor(runtime, budget=4000))])` | `05_pydantic_ai.py` |
| Google ADK (1.10 and 2.x) | `LlmAgent(before_model_callback=ceng_adk.model_callback(runtime, budget=4000))` | `06_google_adk.py` |
| LangGraph (1.x) | `create_react_agent(model, tools, pre_model_hook=ceng_lg.compression_node(runtime, budget=4000))` | `07_langgraph.py` |

In a real run of the demos a ~460-token history shrank to 40-55 tokens and
the agent still answered a question about a fact in the oldest turn. The
adapters are one small `HistoryAdapter` each; add another framework by
implementing `flatten`, `rebuild` and `starts_turn`.

## Verify, persist, evolve

```python
context.verify("fits", tokens=4000)                       # LLM-free budget check
context.verify("macro_fallacy", question=..., population=..., tree=[...])

smaller.save("ctx/")                  # OKF markdown bundle (Open Knowledge Format)
smaller.save("ctx.json", format="json")
Context.load("ctx/", runtime=runtime) # round-trips messages, report and metadata
```

`ceng.evolution` implements ACE playbooks (Generator, Reflector, Curator) and
`ceng.bench` measures them; see [BENCHMARKS.md](BENCHMARKS.md).

## Configuration

`Runtime` is the single, explicit bundle of services (backend, cache,
tokenizer, observers, price table). There are no global singletons.
`Runtime.from_env()` reads, with strict parsing:

| Variable | Default | Meaning |
|---|---|---|
| `CENG_BACKEND` | `litellm` | `litellm`, `openai`, `vllm` |
| `CENG_MODEL` | `gpt-4o-mini` | Model id |
| `CENG_BASE_URL` | unset | OpenAI-compatible endpoint (`openai` / `litellm` backends) |
| `CENG_OPTIONS` | `{}` | JSON of provider parameters sent with every call, e.g. `{"reasoning_effort": "low"}` |
| `CENG_CACHE_DIR` | `.ceng/cache` | SQLite cache directory; empty disables caching |
| `CENG_TIMEOUT_SECONDS` | `60` | Per-call timeout |
| `CENG_RETRY_ATTEMPTS` | `3` | Attempts for transient failures |
| `CENG_CONCURRENCY` | `8` | Parallel LLM calls |
| `CENG_RATE_LIMIT_PER_SECOND` | unset | Ceiling on LLM calls started per second |
| `CENG_DEADLINE_SECONDS` | unset | Total time allowed for one request across retries |

**Reasoning models** (gpt-oss, DeepSeek, Nemotron, ...) spend part of the
token cap on hidden thinking and may return nothing visible. ceng detects a
truncated reply with (almost) no visible text and retries with a larger cap
(up to 3 times, 4x each); set a low `reasoning_effort` through
`CENG_OPTIONS` to keep calls cheap and fast.

Resilience is built in: retries with jittered backoff for transient errors
only, circuit breaker, timeouts, bounded concurrency, single-flight
de-duplication, cache keys covering model, prompt text, sampling parameters
and strategy version. See [PRODUCTION.md](PRODUCTION.md).

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md). In short: frozen dataclasses, an ABC and
registry for every extension point, async core with one sync bridge, strict
types, and **no underscore-prefixed names anywhere** (enforced by a test).

## Development

```bash
make setup && make check     # ruff, mypy --strict, pytest (90% coverage gate)
```

Unit tests need no network; deterministic fault injection (retries, timeouts,
circuit breaker) lives in `tests/faults.py` and is **not** shipped. Everything
that depends on model behaviour is tested against a real provider in
`tests/integration/` and skipped unless a key is set:

```bash
export NVIDIA_API_KEY=...          # or CENG_TEST_API_KEY, any OpenAI-compatible key
pytest tests/integration           # CENG_TEST_BASE_URL / CENG_TEST_MODEL override the target
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Examples in [`examples/`](examples/) are
executed by the test suites (`02_llm_free.py` offline, the rest against a
real model).

## References

- Wolf et al., 2026: Partition-Prompt-Aggregate.
- Zhang et al., [arXiv:2510.04618](https://arxiv.org/abs/2510.04618): ACE.
- [arXiv:2510.26493](https://arxiv.org/abs/2510.26493): Context Engineering 2.0 (OKF).

Apache-2.0.
