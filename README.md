# ceng

> Package name: **`ceng`** — GitHub repo: **`sachncs/context`**.

Context engineering for LLMs via **Partition, Prompt, Aggregate** (Wolf
et al., 2026) plus the four patterns the surrounding research
establishes as load-bearing:

- **ACE** (Zhang et al., arXiv:2510.04618) — evolving itemized
  playbooks updated by a Generator → Reflector → Curator loop, with
  verbatim prompts and the paper's recommended defaults.
- **Anthropic** — finite context as a precious resource; U-shape
  preservation in compaction; NOTES.md-style external memory; just-
  in-time retrieval via lightweight identifiers.
- **Coyle / Medium** — U-shaped retention curve; primacy + recency;
  LLM-supported structured note-taking.
- **iwoszapar research roundup** — static AGENTS.md alone barely
  moves the needle; dynamic context compounds. Build feedback
  loops; treat memory as a first-class engineering concern.

`ceng` is the implementation: a small, dependency-light library
(only `litellm` is required at runtime; `pyyaml`, `tiktoken` are
used) that ships:

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
    cache_dir=".ceng_cache",
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
ev = Evolver(llm="gpt-4o-mini", cache_dir=".ceng_cache")
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
    cache_dir=".ceng_cache",
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
    cache_dir=".ceng_cache",
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
    backends.py        # LiteLLM, vLLM, OpenAI adapters
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
