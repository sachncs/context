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
pip install ceng[tokenize]      # tiktoken-powered budgeting
pip install ceng[all]            # everything
```

## Quick start

```python
import ceng
from ceng import ppa_compress

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
```

Then pass `small` straight to your chat-completion call.

## Self-consistency check

```python
from ceng import ppa_check

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

print(verdict.population_estimate, verdict.aggregated_estimate, verdict.self_consistent)
```

`self_consistent=False` means the model fell for the **macro fallacy** — direct population-level answer differs from the leaf-aggregated answer by more than the configured tolerance.

## Backends

```python
import ceng
ceng.set_backend("litellm")      # default; covers vllm OpenAI-compat server
ceng.set_backend("openai")       # raw openai.OpenAI client
ceng.set_backend("vllm")         # in-process vllm.LLM (needs CUDA)
```

Honoured env vars: `CENG_BACKEND`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_BASE`.

## Caching

All LLM calls are memoised on disk in a sqlite database at `cache_dir` (default `.ceng_cache/`). Two namespaces:

- `summarize` — leaf summaries, keyed by `(model, template_version, leaf_text_hash)`.
- `ppa_check` — leaf answers, keyed by `(model, template_version, leaf_question_hash)`.

Deletes + re-runs are free. Different `cache_dir` per project keeps things tidy.

## License

Apache-2.0.

## References

Wolf, P., Kleine Buening, T., Krause, A., & Mendler-Dünner, C. (2026).
*Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models.*
arXiv:2607.15277.
