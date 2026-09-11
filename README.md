# ceng

> Partition-Prompt-Aggregate context engineering for LLMs. Apache-2.0.
> Drop-in `ppa_compress`, `ppa_check`, ACE `Evolver`, and OKF writer/reader.

> Package name: **`ceng`** — GitHub repo: **`sachncs/context`**.

`ceng` is a small, dependency-light Python library that bundles four
production-shaped context-engineering primitives: PPA compression,
ACE-style evolving playbooks, OKF bundles, and U-shape chat
compaction. One `pip install`, no framework lock-in.

## What's in the box

- **`ppa_compress`** — the original partition-prompt-aggregate
  compressor for the longest user message in a chat.
- **`ppa_compress_to_okf`** — writes the compression output as an
  OKF bundle. **By default, only `index.md` + `combined.md` are
  materialised on disk** (Anthropic's just-in-time pattern); pass
  `index_only=False` for the v0.3.0 full-bundle behaviour.
- **`ppa_check`** — the macro-fallacy detector from the original PPA
  paper. Asks the question at the population level and per leaf
  of a binary tree, then compares the answers.
- **`ceng.compact.compact_messages`** — long chat history compaction
  that preserves primacy + recency (U-shape).
- **`ceng.notes.NotesManager`** — filesystem-backed agentic NOTE
  store with atomic writes, LRU compact, and pinned notes.
- **`ceng.playbook.Evolver`** — ACE's Generator → Reflector → Curator
  loop, with the paper's recommended hyperparameters (5 rounds, 0.90
  dedup, 80k token budget).

All entry points are synchronous; no streaming, no async, no
sub-agent orchestration. The library aims to be a primitive, not a
framework.

## Why ceng?

The four primitives above are stitched together from the strongest
context-engineering papers of the last two years:

- **PPA / Partition, Prompt, Aggregate** (Wolf et al.,
  arXiv:2607.15277) — the macro-fallacy-resistant summarisation
  path that partitions a long context, summarises each chunk in
  isolation, then aggregates. Implemented by `ppa_compress`.
- **ACE / Agentic Context Engineering** (Zhang et al.,
  arXiv:2510.04618) — evolving itemised playbooks updated by a
  Generator → Reflector → Curator loop. Implemented by
  `ceng.playbook.Evolver` with the verbatim prompts from the
  upstream repo and the paper's recommended defaults.
- **Anthropic's "Effective context engineering for AI agents"** —
  finite context as a precious resource; U-shape preservation in
  compaction; NOTES.md-style external memory; just-in-time
  retrieval via lightweight identifiers. Implemented by
  `ceng.compact.compact_messages`, `ceng.notes.NotesManager`, and
  `ceng.ppa_compress_to_okf(index_only=True)`.
- **Coyle / Medium write-up** — U-shaped retention curve; primacy
  + recency; LLM-supported structured note-taking. Drives the
  compaction defaults.
- **Context Engineering 2.0** (Hua et al., arXiv:2510.26493) —
  the broader framework that motivates treating memory as a
  first-class engineering concern.
- **OKF / Open Knowledge Format** (Google Cloud, McVeety &
  Hormati, 2026) — the on-disk bundle format used by
  `ceng.ppa_compress_to_okf` and the OKF reader/writer.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Examples](#examples)
- [ACE playbook (evolving context)](#ace-playbook-evolving-context)
- [OKF bundles (just-in-time retrieval)](#okf-bundles-just-in-time-retrieval)
- [Long chat compaction](#long-chat-compaction)
- [External NOTES](#external-notes)
- [Backends](#backends)
- [Architecture](#architecture)
- [Layout](#layout)
- [License](#license)
- [References](#references)

## Examples

Runnable end-to-end scripts in [`examples/`](./examples):

- [`examples/01_quickstart.py`](./examples/01_quickstart.py) —
  `ppa_compress` over a sample document.
- [`examples/02_okf_bundle.py`](./examples/02_okf_bundle.py) —
  `ppa_compress_to_okf` writing a bundle on disk.
- [`examples/03_compact_chat.py`](./examples/03_compact_chat.py) —
  `compact_messages` on a synthetic long chat.
- [`examples/04_evolver_smoke.py`](./examples/04_evolver_smoke.py) —
  `Evolver.run` over three samples demonstrating the
  Generator/Reflector/Curator loop.

Each script falls back to a stub backend when no
`OPENAI_API_KEY` is set, so the directory is runnable in any
sandbox.

## Pipelines at a glance

```text
                ┌───────────────────────────┐
                │  messages (chat history)  │
                └─────────────┬─────────────┘
                              ▼
       ┌────────────────────────────────────────────┐
       │ pick longest user message → target_text    │
       └─────────────┬──────────────────────────────┘
                     ▼
            ┌──────────────────┐
            │  partition_text  │  → leaves[]
            └────────┬─────────┘
                     ▼
       ┌────────────────────────────┐
       │  summarise_leaf (per leaf)  │ ← cache (ceng/cache)
       └────────┬───────────────────┘
                ▼
       ┌──────────────────────────┐
       │  combine_summaries       │  → final_text
       └────────┬─────────────────┘
                ▼
       ┌──────────────────────────────┐
       │  CompressionBundle / OKF     │
       └──────────────────────────────┘
```

Three pipelines ship today:

| Pipeline | Entry point | What it does |
|---|---|---|
| PPA compress | `ceng.ppa_compress` | Long message → short message via partition / summarise / combine. |
| OKF bundle | `ceng.ppa_compress_to_okf` | Same as above, but writes an OKF bundle to disk for just-in-time retrieval. |
| Long-chat compaction | `ceng.compact.compact_messages` | Drop middle, keep head + tail (U-shape). |
| ACE Evolver | `ceng.playbook.Evolver` | Generator → Reflector → Curator loop, with ADD/UPDATE/MERGE/DELETE operations. |

## Install

```bash
pip install ceng                  # litellm + pyyaml
pip install ceng[tokenize]        # + tiktoken
pip install ceng[vllm]            # + vllm (in-process)
pip install ceng[openai]          # + openai
pip install ceng[all]             # everything
pip install -e .[dev]              # local checkout with pytest
```

## Quick start

```python
import ceng
from ceng import ppa_compress, ppa_check

# --- Compression --------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are a careful analyst."},
    {"role": "user",   "content": "<insert very long context here>"},
]

small = ppa_compress(
    messages,
    budget_tokens=2000,
    llm="gpt-4o-mini",       # any litellm model id, incl. vllm-served
    cache_dir=".ceng/cache",
)

# --- Self-consistency check ----------------------------------------------------
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

print(verdict.population_estimate,
      verdict.aggregated_estimate,
      verdict.self_consistent)  # False ⇒ macro fallacy
```

## ACE playbook (evolving context)

The strongest published result in the context-engineering literature
is the **Agentic Context Engineering** (ACE) paper. ceng ships the
verbatim Generator / Reflector / Curator prompts from the official
GitHub repo (`github.com/ace-agent/ace`) and the paper-recommended
hyperparameters:

```python
from ceng.playbook import empty_playbook, Playbook
from ceng.playbook.evolver import Evolver

playbook = empty_playbook()
ev = Evolver(llm="gpt-4o-mini", cache_dir=".ceng/cache")
playbook, stats = ev.run(
    playbook=playbook,
    queries=train_samples,                # [{"question": ..., "ground_truth": ...}, ...]
    evaluator=lambda q, a, s: "correct" if a == s["ground_truth"] else f"wrong: {a!r}",
    max_iterations=5,
)
# playbook.bullets is now full of id: Bullet, content, helpful_count, harmful_count
```

Defaults are paper §A.6 best: `max_reflector_rounds=5`,
`dedup_threshold=0.90`, `playbook_token_budget=80_000`,
`curator_frequency=1`.

## OKF bundles (just-in-time retrieval)

```python
from ceng.compress import ppa_compress_to_okf

# By default only index.md + combined.md are written; per-leaf files
# are not materialised. An agent that wants leaf-N.md reads it
# directly via ceng.okf.read_concept_file when needed.
concepts = ppa_compress_to_okf(
    messages,
    bundle_dir="ctx-out",
    bundle_name="my-context",
    budget_tokens=2000,
    llm="gpt-4o-mini",
    cache_dir=".ceng/cache",
)
```

## Long chat compaction

```python
from ceng.compact import compact_messages

small, prov = compact_messages(
    long_chat_history,
    backend=None,             # use the active backend
    llm="gpt-4o-mini",
    preserve_first=2,         # keep system + first user
    preserve_last=4,          # keep most recent assistant/user
    summarise_middle=True,    # one backend call to summarise the strip
    cache_dir=".ceng/cache",
)
# prov.preserved_first, prov.preserved_last, prov.summarised_count
# all available for logging.
```

The system-role message is never dropped or summarised — a
`ValueError` is raised if the user passed `preserve_first` too low to
keep it in the head.

## External NOTES

```python
from ceng.notes import NotesManager

notes = NotesManager(root=".ceng_notes")
notes.write("team-handbook.md", "# Handbook\nbe kind\n", tags=("pinned",))
notes.append("standup-2026-07-20.md", "\n- ship v0.4\n")
```

## Backends

```python
import ceng
ceng.set_backend("litellm")      # default; covers vllm OpenAI-compat server
ceng.set_backend("openai")       # raw openai.OpenAI client
ceng.set_backend("vllm")         # in-process vllm.LLM
```

Honoured env vars: `CENG_BACKEND`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_BASE`. The third-party SDKs are imported lazily inside
`complete()`.

## Layout

```
src/ceng/
    __init__.py        # top-level exports
    backends.py        # LiteLLM, vLLM, OpenAI adapters + retry/backoff
    cache.py           # sqlite-backed cache with WAL checkpoint
    check.py           # ppa_check (macro-fallacy probe)
    compact/           # ceng.compress package
        __init__.py    # ppa_compress, ppa_compress_to_okf, ...
        prompts.py     # leaf-summarise + combine prompts
        bundle.py      # CompressionBundle -> OKF concepts
        log.py         # ceng logger
    compact.py         # chat-history compaction (U-shape)
    compress/          # re-export shim for backward compat
    notes.py           # agentic NOTE-style external memory
    okf.py             # OKF v0.1 reader/writer
    partition.py       # adaptive text partitioner
    playbook/          # ACE-style evolving bullet playbook
        __init__.py    # Bullet, Playbook, parse/render/merge
        prompts.py     # verbatim ACE prompts
        evolver.py     # Generator -> Reflector -> Curator loop
    tokens.py          # tiktoken + heuristic token counter
tests/                # 260+ tests, fixture-driven, no live LLMs
```

## Architecture

The full architecture document (module map, cache topology, backend
trade-offs, extension points) lives in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## License

Apache-2.0.

## References

- Wolf, P., Kleine Buening, T., Krause, A., & Mendler-Dünner, C. (2026).
  *Partition, Prompt, Aggregate: Statistical Self-Consistency in
  Language Models.* arXiv:2607.15277.
- Zhang, Q., Hu, C., Upasani, S., et al. (2026). *Agentic Context
  Engineering: Evolving Contexts for Self-Improving Language Models.*
  arXiv:2510.04618. Code: <https://github.com/ace-agent/ace>.
- Hua, Q., Ye, L., Fu, D., et al. (2025). *Context Engineering 2.0: The
  Context of Context Engineering.* arXiv:2510.26493.
- Anthropic. (2025). *Effective context engineering for AI agents.*
  <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>.
- McVeety, S., & Hormati, A. (2026). *Introducing the Open Knowledge
  Format.* Google Cloud Blog.
  <https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing>.
- Coyle, F. (2025). *Context Engineering: The New AI Frontier.*
  Medium.
- Szapar, I. (2026). *Context Engineering Research: 2026 Papers and
  Benchmarks.* iwoszapar.com.
