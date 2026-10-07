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
| `CENG_CACHE_DIR` | `.ceng/cache` | SQLite cache directory; empty disables caching |
| `CENG_TIMEOUT_SECONDS` | `60` | Per-call timeout |
| `CENG_RETRY_ATTEMPTS` | `3` | Attempts for transient failures |
| `CENG_CONCURRENCY` | `8` | Parallel LLM calls |
| `CENG_RATE_LIMIT_PER_SECOND` | unset | Ceiling on LLM calls started per second |
| `CENG_DEADLINE_SECONDS` | unset | Total time allowed for one request across retries |

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

See [CONTRIBUTING.md](CONTRIBUTING.md). Examples in [`examples/`](examples/) run
offline and are executed by the test suite.

## References

- Wolf et al., 2026: Partition-Prompt-Aggregate.
- Zhang et al., [arXiv:2510.04618](https://arxiv.org/abs/2510.04618): ACE.
- [arXiv:2510.26493](https://arxiv.org/abs/2510.26493): Context Engineering 2.0 (OKF).

Apache-2.0.
