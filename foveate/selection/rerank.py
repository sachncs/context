"""LLM re-ranking of a first-stage retriever's candidates."""

from __future__ import annotations

import logging

from foveate import errors, prompts
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.internals import jsonout
from foveate.selection import base

logger = logging.getLogger("foveate.selection")

RERANK = prompts.PromptTemplate(
    name="selection.rerank",
    version="1",
    system=(
        "You rank passages by how useful they are for answering a question. "
        "Reply with JSON only."
    ),
    user=(
        "Question: {question}\n\nPassages (id: text):\n{passages}\n\n"
        'Return {{"ranking": [ids, most useful first]}} listing only passages '
        "that help answer the question."
    ),
)
SNIPPET_CHARS = 700


@base.Retriever.registry.register("rerank")
class LlmReranker(base.Retriever):
    """Asks the model to reorder the first-stage candidates.

    If the model's reply is unusable the first-stage order is kept, so
    re-ranking can only improve or leave the result unchanged.
    """

    def __init__(
        self,
        first_stage: base.Retriever,
        runtime: runtime_lib.Runtime,
        candidates: int = 20,
    ) -> None:
        """Creates the re-ranker.

        Raises:
            ConfigError: If `candidates` is below two.
        """
        if candidates < 2:
            raise errors.ConfigError("candidates must be >= 2")
        self.first_stage = first_stage
        self.runtime = runtime
        self.candidates = candidates

    async def search(self, query: str, k: int) -> list[base.Hit]:
        pool = await self.first_stage.search(query, max(k, self.candidates))
        if len(pool) < 2:
            return pool[:k]
        listing = "\n".join(
            f"{h.chunk.id}: {h.chunk.text[:SNIPPET_CHARS]!r}" for h in pool
        )
        try:
            completion = await self.runtime.complete(
                (
                    messages_lib.Message(
                        messages_lib.Role.SYSTEM, RERANK.system
                    ),
                    messages_lib.Message(
                        messages_lib.Role.USER,
                        RERANK.render_user(question=query, passages=listing),
                    ),
                ),
                source="rerank",
                namespace=f"rerank:{RERANK.fingerprint}",
                max_tokens=600,
            )
            ranking = jsonout.extract_json(completion.text).get("ranking")
        except (errors.BackendError, errors.ValidationError) as exc:
            logger.warning("rerank failed, keeping first-stage order: %s", exc)
            return pool[:k]
        order = [str(i) for i in ranking] if isinstance(ranking, list) else []
        by_id = {h.chunk.id: h for h in pool}
        chosen = [by_id[i] for i in dict.fromkeys(order) if i in by_id]
        rest = [h for h in pool if h.chunk.id not in set(order)]
        total = len(pool)
        return [
            base.Hit(h.chunk, float(total - rank))
            for rank, h in enumerate(chosen + rest)
        ][:k]

    async def aclose(self) -> None:
        await self.first_stage.aclose()
