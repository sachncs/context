---
title: "Extending Foveate"
description: "Every extension point is an abstract class with a registry: add a compression method, loader, retriever, backend, cache or observer."
---

Each extension point is an abstract base class. Registering is a decorator, and an unknown name raises
`ConfigError` listing the known ones.

| To add | Subclass | Register with |
|---|---|---|
| A compression method | `compression.Compressor` | `@Compressor.register("name")` |
| A tool-output format | `compression.Reducer` | `@Reducer.registry.register("name")` |
| A document format | `documents.Loader` | `@Loader.registry.register("name")` |
| A retriever | `selection.Retriever` | `@Retriever.registry.register("name")` |
| An embedder | `backends.embeddings.Embedder` | pass to `Runtime(embedder=...)` |
| A benchmark pipeline | `bench.longdoc.Pipeline` | `@Pipeline.registry.register("name")` |
| A model backend | `backends.Backend` | pass to `Runtime(backend=...)` |
| A cache | `cache.Cache` | pass to `Runtime(cache=...)` |
| An observer | `observability.Observer` | `Runtime(observers=(...))` |
| An agent-framework adapter | `agents.HistoryAdapter` | see [Agents and tools](../guides/agents-and-tools.md) |

## A compression method in twenty lines

```python
import dataclasses
from foveate.compression import Compressor

@Compressor.register("first_sentences")
@dataclasses.dataclass(frozen=True)
class FirstSentences(Compressor):
    """Keeps the first sentence of every message."""

    async def run(self, context, budget, trace):
        messages = tuple(
            m.with_content(m.content.split(". ")[0] + ".") for m in context.messages
        )
        return dataclasses.replace(context, messages=messages)

smaller = context.compress("first_sentences", budget=500)
```

You implement `run`; the base class handles the rest of the contract: if the context already fits it is
returned unchanged, otherwise `run` is called, the budget is enforced (raising `BudgetExceededError` or
truncating, as the caller chose), and a `CompressionReport` is attached. Options are the dataclass fields,
so they are validated, hashable and visible in `repr`.

## Conventions

* Value types are frozen dataclasses.
* Every model call goes through `Runtime.complete`.
* No underscore-prefixed names: what is public is what a package exports in `__all__`.
* Unit tests are offline and deterministic; behaviour that depends on a real model belongs in the
  integration tests, which are skipped without a key.
