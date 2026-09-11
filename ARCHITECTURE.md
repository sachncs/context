# Architecture

This document is the maintainer's map of ceng: where each module
fits, how the four public pipelines share resources, and where to
extend the package. For per-function reference see the docstrings;
for runnable usage see the [README](./README.md) and
[`examples/`](./examples).

## Module map

Under [`src/ceng/`](./src/ceng):

| Module | Responsibility | Depends on |
|---|---|---|
| `__init__.py` | Top-level exports (`ppa_compress`, `ppa_check`, `set_backend`, OKF primitives, ...) and `__version__`. Re-exports `configure_logging` from `compress.log` so callers can do `ceng.configure_logging(level=logging.INFO)` once at startup. | everything below |
| `backends.py` | `Backend` Protocol and three adapters (`LiteLLMBackend`, `VLLMBackend`, `OpenAIBackend`) behind `set_backend(name)` selector. Also hosts `retry_with_backoff` for transient 5xx / rate-limit recovery. | `litellm`, `vllm`, `openai` (lazy) |
| `cache.py` | `Cache` — sqlite-on-disk KV store with WAL mode, `__enter__`/`__exit__`, `PRAGMA busy_timeout`, `PRAGMA user_version` migration. Two namespaces (`summarize`, `ppa_check`). | stdlib only |
| `check.py` | `ppa_check` — macro-fallacy probe. Iteratively queries the model at the population level and at each leaf of a binary tree, then compares answers. Validates the tree (`prior == 0` rejection, scientific notation parsing, iterative flatten). | `backends`, `cache`, `partition`, `tokens` |
| `compress/` | The PPA compression package. Public via `ceng.compress` and re-exported at top level. | `backends`, `cache`, `okf`, `partition`, `tokens` |
| `compress/__init__.py` | `ppa_compress`, `compress_to_bundle`, `ppa_compress_to_okf`. The leaf-summary / combine / replace pipeline. Emits INFO logs. | `compress.bundle`, `compress.log`, `compress.prompts` |
| `compress/bundle.py` | `CompressionBundle` → OKF `Concept[]` mapping. Combined-concept, leaf-concept, index-concept, no-op index-concept builders. | `okf` |
| `compress/log.py` | `ceng` logger + `configure_logging(level)` opt-in helper. | stdlib only |
| `compress/prompts.py` | The leaf-summarise and combine prompts and prompt-builder functions. `PROMPT_VERSION` bumps invalidate the cache key. | stdlib only |
| `compact.py` | `compact_messages` — U-shape compaction that keeps head + tail verbatim and summarises the middle in one backend call. Refuses to drop a system-role message. | `backends`, `cache`, `tokens` |
| `eval/` | The ACE-shaped benchmark harness. | `backends`, `cache` |
| `eval/__init__.py` | `DataProcessor` Protocol, `DataSample`, `EvalResult`, `run_eval`, `render_report`, `write_report`. Counts `backend_errors` separately from wrong predictions. | stdlib only |
| `eval/{finer,formula,ddxplus,appworld}.py` | Per-benchmark processors and (where applicable) seed playbooks. | `playbook`, `playbooks` |
| `notes.py` | `NotesManager` — filesystem-backed NOTE store with atomic writes, LRU compact, pinned-`_` prefix. | stdlib only |
| `okf.py` | OKF v0.1 reader / writer. `Concept`, `Frontmatter`, `read_bundle`, `write_bundle`, `find_concept`, progressive-disclosure helpers. | `pyyaml` |
| `partition.py` | Adaptive text partitioner — sentence → paragraph → word → greedy word split. Hard-caps individual words at `WORD_MAX_BYTES`. | `tokens` |
| `playbook/` | The ACE evolving-bullet playbook package. | `cache`, `tokens` |
| `playbook/__init__.py` | `Bullet`, `Playbook`, `parse_playbook`, `render_playbook`, `merge`, `trim_to_token_budget`. | `tokens` |
| `playbook/prompts.py` | Verbatim ACE Generator / Reflector / Curator prompts and message builders. | stdlib only |
| `playbook/evolver.py` | The Generator → Reflector → Curator loop. Supports ADD / UPDATE / MERGE / DELETE Curator operations. | `backends`, `cache`, `playbook`, `playbook.prompts` |
| `playbooks/__init__.py` | Curated seed playbooks for FiNER / Formula / DDXPlus / AppWorld (markdown strings). | stdlib only |
| `presets.py` | Named `EvolverConfig` instances mirroring the paper's hyperparameter sweeps. | stdlib only |
| `tokens.py` | `count_tokens` — tiktoken (when installed) or `len(text) // 4` heuristic. `token_budget_split` for sentence-aware budgeting. | `tiktoken` (optional) |
| `bench.py` | The `ceng bench <benchmark>` CLI and the `finer` / `formula` / `ddxplus` / `appworld` / `smoke` / `all` runner functions. `appworld` raises `NotImplementedError` until the gated dataset is wired. | `backends`, `eval` |
| `py.typed` | PEP 561 marker so downstream `mypy --strict` sees the inline annotations. | n/a |

## Pipeline diagram

```text
                       ┌───────────────────────────────┐
                       │  messages: chat message list  │
                       └───────────────┬───────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
  ┌────────────────────┐                              ┌────────────────────┐
  │  ppa_compress      │                              │  compact_messages  │
  │  (longest user msg)│                              │  (U-shape keep)    │
  └─────────┬──────────┘                              └─────────┬──────────┘
            ▼                                                     ▼
  ┌────────────────────┐                              ┌────────────────────┐
  │  partition_text    │                              │  summarise middle  │
  └─────────┬──────────┘                              │  (one LLM call)    │
            ▼                                         └─────────┬──────────┘
  ┌────────────────────┐                                        │
  │  summarise_leaf ×N │                                        │
  │  (cache by sha256) │                                        │
  └─────────┬──────────┘                                        │
            ▼                                                     │
  ┌────────────────────┐                                          │
  │  combine_summaries │                                          │
  └─────────┬──────────┘                                          │
            ▼                                                     ▼
  ┌────────────────────┐                              ┌────────────────────┐
  │  CompressionBundle │                              │  CompactionResult  │
  │  or OKF bundle     │                              │  + Provenance      │
  └────────────────────┘                              └────────────────────┘

  ACE Evolver (playbook.evolver):

  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │  Generator         │─▶│  Reflector         │─▶│  Curator           │
  │  (use playbook)    │  │  (only on FAILED)  │  │  ADD/UPDATE/       │
  └────────────────────┘  └────────────────────┘  │  MERGE/DELETE      │
                                                  └─────────┬──────────┘
                                                            ▼
                                                  ┌────────────────────┐
                                                  │  Playbook.merge    │
                                                  │  (content-hash     │
                                                  │   dedup, 0.90)     │
                                                  └────────────────────┘
```

## Cache topology

All public functions default to the same on-disk root:
`.ceng/cache/` (sqlite file: `cache.sqlite`). Two namespaces live
inside it:

* `"summarize"` — leaf summaries, combined summaries, and
  Generator/Reflector/Curator responses from the Evolver.
* `"ppa_check"` — population-level and leaf-level answers from
  `ppa_check`.

Key derivation is `:func:ceng.cache.make_key` — SHA-256 over a
canonical JSON dump of the prompt components, so two callers
producing the same inputs always hit the same row.

To share a cache across two pipelines, pass the same `cache_dir=`
explicitly:

```python
ppa_compress(messages, budget_tokens=2000, llm="gpt-4o-mini",
             cache_dir=".ceng/cache")
compact_messages(messages, llm="gpt-4o-mini",
                 cache_dir=".ceng/cache")
```

To isolate (e.g. across two concurrent Evolver runs), pass two
distinct paths.

## Backend abstraction

Three adapters ship in `ceng.backends`:

| Backend | When to pick it | Trade-offs |
|---|---|---|
| `litellm` (default) | One code path across hosted vLLM (OpenAI-compatible HTTP), OpenAI, Anthropic, Bedrock, etc. | Adds `litellm` as a runtime dependency. Retries transient failures with exponential backoff + jitter. |
| `openai` | Raw `openai.OpenAI` client; no `litellm`. | Smaller dependency surface. Same retry semantics. |
| `vllm` | In-process `vllm.LLM` for users running vLLM directly on a GPU. | No retry — in-process generation failures are usually programmer error. Requires CUDA + the `vllm` extra. |

All three are wrapped by `retry_with_backoff` (`CENG_RETRY`,
`CENG_RETRY_BASE_MS` env vars). The retry count and base delay
are tunable per-backend (`retry_attempts`, `retry_base_ms`) or
globally via env vars.

## Extension points

### Add a new backend

Subclass `ceng.backends.Backend` and add it to `_BACKEND_REGISTRY`
in `ceng/backends.py`. Two methods to implement:

* `name: str` — the registry key.
* `complete(messages, model, **kw) -> str` — send the chat
  messages to the model and return the assistant text.

Wrap `complete()` in `retry_with_backoff` if you want transient
failures to recover.

### Add a new ACE preset

Append a new `EvolverConfig` factory to `ceng/presets.py`. Use the
existing `EvolverConfig` dataclass; the preset just sets the
hyperparameters you want.

### Add a new benchmark processor

Implement the `DataProcessor` Protocol in `ceng.eval`:

```python
from ceng.eval import DataProcessor, DataSample

class MyProcessor:
    def process_task_data(self, raw_data):
        return [DataSample(question=r["q"], target=r["a"]) for r in raw_data]
    def answer_is_correct(self, predicted, ground_truth):
        return predicted.strip().lower() == ground_truth.strip().lower()
    def evaluate_accuracy(self, predictions, ground_truths):
        correct = sum(self.answer_is_correct(p, g)
                      for p, g in zip(predictions, ground_truths))
        return correct / len(predictions) if predictions else 0.0
```

Then wire it into `ceng.bench` if you want a CLI runner.

### Add a new OKF concept type

1. Add a constant string in `ceng.okf` (e.g. `CENG_NEW_TYPE = "new_type"`).
2. Use it in `Frontmatter(type=CENG_NEW_TYPE)` when building the
   `Concept`.
3. Add a progressive-disclosure helper if you want callers to be
   able to filter on it.

### Add a new ACE Curator operation

Extend `_apply_curator_operations` in `ceng/playbook/evolver.py`
with a new branch on `op.get("type")`. Each branch mutates the
playbook in place and returns a fragment for the `excerpt` field
so stats report what the Curator did.

## Logging

`ceng.compress.log.logger` is the shared logger (name `"ceng"`).
Callers opt in with:

```python
import logging
import ceng
ceng.configure_logging(level=logging.INFO)
```

The compression pipeline emits five INFO lines per call:
`ppa_compress start`, `ppa_compress partitioned`, and
`ppa_compress done` — plus per-leaf / per-combine log lines if you
turn on DEBUG. The default level is `WARNING`, so silent-by-default
behaviour is preserved.
