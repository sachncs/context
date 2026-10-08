"""Reciprocal-rank fusion of several retrievers."""

from __future__ import annotations

from collections.abc import Sequence

from foveate import errors
from foveate.selection import base

OVERSAMPLE = 4


@base.Retriever.registry.register("hybrid")
class HybridRetriever(base.Retriever):
    """Fuses rankings with weighted reciprocal-rank fusion (RRF).

    RRF needs no score calibration: a chunk's fused score is the sum over
    retrievers of `weight / (rrf_k + rank)`.
    """

    def __init__(
        self,
        retrievers: Sequence[base.Retriever],
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        """Creates the fusion.

        Raises:
            ConfigError: For no retrievers, mismatched or non-positive
                weights, or `rrf_k` < 1.
        """
        if not retrievers:
            raise errors.ConfigError("hybrid needs at least one retriever")
        use = list(weights) if weights is not None else [1.0] * len(retrievers)
        if len(use) != len(retrievers) or any(w <= 0 for w in use):
            raise errors.ConfigError("need one positive weight per retriever")
        if rrf_k < 1:
            raise errors.ConfigError("rrf_k must be >= 1")
        self.retrievers = tuple(retrievers)
        self.weights = tuple(use)
        self.rrf_k = rrf_k

    async def search(self, query: str, k: int) -> list[base.Hit]:
        fused: dict[str, float] = {}
        chunks = {}
        for retriever, weight in zip(
            self.retrievers, self.weights, strict=True
        ):
            hits = await retriever.search(query, k * OVERSAMPLE)
            for rank, hit in enumerate(hits, start=1):
                chunks[hit.chunk.id] = hit.chunk
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight / (
                    self.rrf_k + rank
                )
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [base.Hit(chunks[i], s) for i, s in ranked[:k]]

    async def aclose(self) -> None:
        for retriever in self.retrievers:
            await retriever.aclose()
