# ceng

Context engineering for LLMs via **Partition, Prompt, Aggregate** (Wolf et al., 2026).

Two utilities, one cache, three pluggable backends:

- **`ppa_compress`** — replace a bloated message with a compressed, higher-fidelity one by aggregating leaf summaries instead of doing one lossy whole-document pass.
- **`ppa_check`** — run the paper's binary-tree self-consistency probe on any population-level question and detect the *macro fallacy*.

Works with vLLM (OpenAI-compatible server or in-process), OpenAI, Anthropic, and any other provider that `litellm` speaks.

## Install

```bash
pip install ceng                 # litellm backend (default)
pip install ceng[vllm]           # in-process vllm.LLM backend
pip install ceng[openai]         # raw openai client
pip install ceng[tokenize]       # tiktoken-powered budgeting
pip install ceng[dev]            # pytest, etc.
pip install ceng[all]            # everything
```

## How it works

### Compression

```
                 ┌─────────────────────────────────────┐
   big input ──▶ │ partition_text into leaf chunks      │
                 └─────────────────────────────────────┘
                              │
                ┌─────────────┴────────────┐
                ▼                          ▼
        summarise leaf₁ …          summarise leafₙ     (cached, 1 call/leaf)
                │                          │
                └─────────────┬────────────┘
                              ▼
                     combine summaries       (1 call)
                              ▼
                  ◀── single coherent summary
```

The naive "summarise the whole document in one shot" path is the
**macro fallacy** — the model knows the parts but can't sum them. The
PPA path asks the model about each chunk and aggregates.

### Consistency check

```
       question about population ──▶ P₀  (direct estimate)

       question about every leaf  ──▶ {pᵢ}  with priors {wᵢ}
                                    ▼
                         P₁ = Σ wᵢ · pᵢ   (aggregated estimate)

       |P₀ − P₁| ≤ tolerance ?    ──▶  self_consistent: bool
```

A `False` verdict means the model fell for the macro fallacy on the
given question — the population-level answer it gave is worse than
the prior-weighted leaf aggregate.

## Quick start

```python
import ceng
from ceng import ppa_compress, ppa_check

# --- Compression ------------------------------------------------------
messages = [
    {"role": "system", "content": "You are a careful analyst."},
    {"role": "user",   "content": "<insert very long context here>"},
]

small = ppa_compress(
    messages,
    budget_tokens=2000,
    llm="gpt-4o-mini",       # any litellm model id, incl. vllm-served
    cache_dir=".ceng_cache",
)

# Use `small` straight in your chat-completion call.

# --- Self-consistency check ------------------------------------------
verdict = ppa_check(
    question="What fraction of our users prefer feature X?",
    population="All of our users",
    tree=[
        {"description": "EU users",      "prior": 0.3, "children": [
            {"description": "German users", "prior": 0.5},
            {"description": "French users", "prior": 0.5},
        ]},
        {"description": "US users",      "prior": 0.7},
    ],
    llm="gpt-4o-mini",
)

print(
    verdict.population_estimate,
    verdict.aggregated_estimate,
    verdict.self_consistent,  # False ⇒ model committed the macro fallacy
)
```

## Backends

```python
import ceng
ceng.set_backend("litellm")      # default; covers vllm OpenAI-compat server
ceng.set_backend("openai")       # raw openai.OpenAI client
ceng.set_backend("vllm")         # in-process vllm.LLM (needs CUDA)
```

Honoured env vars: `CENG_BACKEND`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_BASE` (the standard `litellm`-style knobs).

The heavy third-party SDKs (`litellm`, `vllm`, `openai`) are imported
lazily inside each `complete` call, so installing just `ceng` is enough
to use the litellm path; the other two paths fail loudly at first use
with an `ImportError` that suggests the matching extra.

## Caching

All LLM calls are memoised on disk in a sqlite database at `cache_dir`
(default `.ceng_cache/`). Two namespaces:

- `summarize` — leaf summaries, keyed by `(model, template_version, leaf_text_sha256)`.
- `ppa_check` — leaf answers, keyed by `(model, template_version, leaf_question_sha256)`.

Re-running is free. Different `cache_dir` per project keeps things tidy.
Pass `cache_dir=""` to disable caching for a single call.

## Development

```bash
pip install -e .[dev]
pytest -q                       # 90+ tests, no live LLM calls
```

Layout:

```
src/ceng/
    cache.py       # sqlite cache
    backends.py    # litellm / vllm / openai adapters
    tokens.py      # tiktoken + heuristic token counters
    partition.py   # content-aware recursive chunker
    compress.py    # ppa_compress
    check.py       # ppa_check
tests/
    test_cache.py
    test_backends.py
    test_tokens.py
    test_partition.py
    test_compress.py
    test_check.py
    test_package.py
```

## License

Apache-2.0.

## References

Wolf, P., Kleine Buening, T., Krause, A., & Mendler-Dünner, C. (2026).
*Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models.*
arXiv:2607.15277.
