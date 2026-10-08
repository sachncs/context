"""Dense retrieval with any `Embedder`, with an on-disk vector cache."""

from __future__ import annotations

import json
from collections.abc import Sequence

from foveate.backends import embeddings
from foveate.cache import base as cache_base
from foveate.documents import chunking
from foveate.internals import hashing
from foveate.selection import base


@base.Retriever.registry.register("embedding")
class EmbeddingRetriever(base.Retriever):
    """Cosine similarity between a query vector and chunk vectors.

    Chunk vectors are computed on first use and cached by (model, text), so
    re-indexing an unchanged document costs nothing.
    """

    def __init__(
        self,
        chunks: Sequence[chunking.Chunk],
        embedder: embeddings.Embedder,
        cache: cache_base.Cache | None = None,
    ) -> None:
        """Stores the chunks; vectors are built lazily.

        Args:
            chunks: Chunks to index.
            embedder: Vector model.
            cache: Optional cache for chunk vectors.
        """
        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.cache = cache
        self.vectors: list[list[float]] | None = None

    def cache_key(self, text: str) -> str:
        """Returns the cache key of one chunk's vector."""
        return hashing.fingerprint("vector", self.embedder.model, text)

    async def build(self) -> list[list[float]]:
        """Embeds every chunk (cache first) and returns the vectors."""
        if self.vectors is not None:
            return self.vectors
        found: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, chunk in enumerate(self.chunks):
            raw = (
                self.cache.get(self.cache_key(chunk.searchable))
                if self.cache
                else None
            )
            if raw is not None:
                try:
                    found[index] = [float(x) for x in json.loads(raw)]
                    continue
                except (ValueError, TypeError):
                    pass
            missing.append(index)
        if missing:
            fresh = await self.embedder.embed(
                [self.chunks[i].searchable for i in missing], embeddings.PASSAGE
            )
            for index, vector in zip(missing, fresh, strict=True):
                found[index] = vector
                if self.cache is not None:
                    self.cache.set(
                        self.cache_key(self.chunks[index].searchable),
                        json.dumps(vector),
                    )
        self.vectors = [found[i] for i in range(len(self.chunks))]
        return self.vectors

    async def search(self, query: str, k: int) -> list[base.Hit]:
        vectors = await self.build()
        (query_vector,) = await self.embedder.embed([query], embeddings.QUERY)
        scored = [
            (sum(a * b for a, b in zip(query_vector, vector, strict=True)), i)
            for i, vector in enumerate(vectors)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [base.Hit(self.chunks[i], s) for s, i in scored[:k] if s > 0]

    async def aclose(self) -> None:
        await self.embedder.aclose()
