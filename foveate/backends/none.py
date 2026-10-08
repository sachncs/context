"""Backend for runtimes that must never call a model."""

from __future__ import annotations

from foveate import errors
from foveate.backends import base


@base.Backend.registry.register("none")
class NoBackend(base.Backend):
    """Rejects every request with a clear error.

    Use it (via `Runtime.without_llm`) for the LLM-free compression methods
    (`window`, `truncate`, `extractive`, `ushape` with `middle="drop"`,
    `offload`), so a misconfigured strategy fails loudly instead of trying
    to reach a provider.
    """

    async def complete(self, request: base.Request) -> base.Completion:
        raise errors.PermanentBackendError(
            "this runtime has no LLM backend; use an LLM-free method or "
            "configure a backend"
        )
