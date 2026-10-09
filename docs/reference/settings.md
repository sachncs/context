---
title: "Settings and environment variables"
description: "Every environment variable Runtime.from_env reads, with its default."
---

`Runtime.from_env()` builds a `Runtime` from these variables. A value that cannot be parsed raises
`ConfigError`; nothing is silently ignored.

| Variable | Default | Meaning |
|---|---|---|
| `FOVEATE_BACKEND` | `openai` | `openai` (any OpenAI-compatible HTTP endpoint) or `vllm` (in-process) |
| `FOVEATE_MODEL` | `gpt-4o-mini` | Model name sent to the server |
| `FOVEATE_BASE_URL` | `OPENAI_BASE_URL`, else OpenAI | The server's `/v1` root |
| `OPENAI_API_KEY` | none | Bearer token for the server |
| `FOVEATE_OPTIONS` | none | JSON object merged into each request, for example `{"reasoning_effort": "low"}` |
| `FOVEATE_CONTEXT_WINDOW` | from a built-in table | Tokens the model accepts; overrides the table |
| `FOVEATE_EMBEDDING_MODEL` | none | Embedding model on the same endpoint, enabling embedding and hybrid retrieval |
| `FOVEATE_CACHE_DIR` | `.foveate/cache` | Where the response cache lives; an empty value disables caching |
| `FOVEATE_TIMEOUT_SECONDS` | `60` | Per-call timeout |
| `FOVEATE_RETRY_ATTEMPTS` | `3` | Attempts per call (rate limits, timeouts, 5xx) |
| `FOVEATE_CONCURRENCY` | `8` | Most simultaneous calls |
| `FOVEATE_RATE_LIMIT_PER_SECOND` | none | Calls per second across the process |
| `FOVEATE_DEADLINE_SECONDS` | none | Total time allowed for one call including retries |
| `FOVEATE_CACHE_HOME` | `~/.cache/foveate` | Where benchmark downloads are stored |

## In code

The same settings are constructor arguments, which is clearer in an application:

```python
from foveate import Runtime
from foveate.backends import OpenAIBackend, ResilientBackend, RetryPolicy

runtime = Runtime(
    backend=ResilientBackend(
        OpenAIBackend(base_url="http://localhost:8000/v1", api_key="local"),
        retry=RetryPolicy(attempts=4, base_delay=1.0, max_delay=30.0),
        timeout=120.0,
        max_concurrency=4,
        rate_per_second=2.0,
    ),
    model="my-model",
    context_window=16384,
    options={"reasoning_effort": "low"},
)
```

## Tokenizers

The default tokenizer estimates about four characters per token. The `foveate[tokenize]` extra (see [Install](../install.md))
adds `tiktoken` and exact counts for OpenAI models. Budgets exclude per-message framing tokens, so
leave a little headroom against the model's hard limit.
