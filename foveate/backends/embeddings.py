"""Embedding backends for semantic retrieval."""

from __future__ import annotations

import abc
import math
import re
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from foveate import errors
from foveate.backends import classify, providers, resilient
from foveate.internals import looplocal, registry

WORD = re.compile(r"[a-z0-9]+")
PASSAGE = "passage"
QUERY = "query"


class Embedder(abc.ABC):
    """Turns text into vectors. Vectors are L2-normalised."""

    registry: ClassVar[registry.Registry[type[Embedder]]] = registry.Registry(
        "embedder"
    )
    model: str = ""

    @abc.abstractmethod
    async def embed(
        self, texts: Sequence[str], kind: str = PASSAGE
    ) -> list[list[float]]:
        """Embeds `texts`.

        Args:
            texts: Strings to embed.
            kind: `"passage"` (documents) or `"query"`; some models embed the
                two differently.

        Returns:
            One unit vector per input, in order.
        """

    async def aclose(self) -> None:
        """Releases resources. Default: nothing to release."""


def normalise(vector: list[float]) -> list[float]:
    """Scales `vector` to unit length (zero vectors are returned as is)."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


@Embedder.registry.register("hashing")
class HashingEmbedder(Embedder):
    """Offline embedder using feature hashing of words (no model needed).

    Captures lexical overlap in a fixed-size dense space. It is not
    semantic, but it is deterministic, dependency-free and good enough to
    rank passages that share vocabulary with the query.

    Attributes:
        dim: Vector dimensionality.
    """

    def __init__(self, dim: int = 512) -> None:
        """Creates the embedder.

        Raises:
            ConfigError: If `dim` is below 8.
        """
        if dim < 8:
            raise errors.ConfigError("dim must be >= 8")
        self.dim = dim
        self.model = f"hashing-{dim}"

    async def embed(
        self, texts: Sequence[str], kind: str = PASSAGE
    ) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in WORD.findall(text.lower()):
                digest = zlib.crc32(word.encode("utf-8"))
                sign = 1.0 if (digest >> 31) & 1 else -1.0
                vector[digest % self.dim] += sign
            vectors.append(normalise(vector))
        return vectors


@Embedder.registry.register("openai")
class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible `/embeddings` endpoint (OpenAI, NVIDIA, vLLM).

    Attributes:
        model: Embedding model id.
        batch_size: Texts per request.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
        passage_options: Mapping[str, object] | None = None,
        query_options: Mapping[str, object] | None = None,
        retry: resilient.RetryPolicy | None = None,
    ) -> None:
        """Stores settings; the client is built lazily per event loop.

        Args:
            model: Embedding model id.
            base_url: Endpoint override.
            api_key: Explicit key; otherwise `OPENAI_API_KEY`.
            batch_size: Texts per request (>= 1).
            passage_options: Extra body for passage embeddings, for example
                `{"input_type": "passage"}` on retrieval models that need it.
            query_options: Extra body for query embeddings.
            retry: Retry policy for transient failures.

        Raises:
            ConfigError: If `batch_size` is below one.
        """
        if batch_size < 1:
            raise errors.ConfigError("batch_size must be >= 1")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.options = {
            PASSAGE: dict(passage_options or {}),
            QUERY: dict(query_options or {}),
        }
        self.retry = retry or resilient.RetryPolicy(attempts=4, base_delay=1.0)
        self.clients: looplocal.LoopLocal[Any] = looplocal.LoopLocal(
            self.build_client
        )

    def build_client(self) -> Any:
        """Builds an async client (once per event loop)."""
        openai = providers.require("openai", "openai")
        return openai.AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, max_retries=0
        )

    async def embed_batch(
        self, texts: Sequence[str], kind: str
    ) -> list[list[float]]:
        """Embeds one batch with retries."""
        client = self.clients.get()

        async def call() -> list[list[float]]:
            try:
                extra = self.options.get(kind) or None
                response = await client.embeddings.create(
                    model=self.model, input=list(texts), extra_body=extra
                )
            except Exception as exc:
                raise classify.classify(exc) from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(texts):
                raise errors.ValidationError("embedding count mismatch")
            return [normalise(list(item.embedding)) for item in ordered]

        return await resilient.retry_async(call, self.retry)

    async def embed(
        self, texts: Sequence[str], kind: str = PASSAGE
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                await self.embed_batch(
                    texts[start : start + self.batch_size], kind
                )
            )
        return vectors

    async def aclose(self) -> None:
        client = self.clients.discard()
        if client is not None:
            await client.close()
