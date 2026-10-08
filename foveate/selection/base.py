"""Retrieval interface: find the chunks (and pages) relevant to a query."""

from __future__ import annotations

import abc
import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from foveate.documents import chunking
from foveate.internals import registry


@dataclasses.dataclass(frozen=True, slots=True)
class Hit:
    """A retrieved chunk with its relevance.

    Attributes:
        chunk: The chunk.
        score: Higher is more relevant; comparable only within one search.
    """

    chunk: chunking.Chunk
    score: float


class Retriever(abc.ABC):
    """Ranks a fixed set of chunks against queries."""

    registry: ClassVar[registry.Registry[type[Retriever]]] = registry.Registry(
        "retriever"
    )

    @abc.abstractmethod
    async def search(self, query: str, k: int) -> list[Hit]:
        """Returns up to `k` hits, best first (never padded with zeros)."""

    async def aclose(self) -> None:
        """Releases resources. Default: nothing to release."""


def page_scores(hits: Sequence[Hit]) -> dict[tuple[str, int], float]:
    """Aggregates chunk hits into `(doc_id, page)` scores.

    A page scores its best chunk plus 10% of every further matching chunk,
    so pages with several relevant passages edge out single-hit pages.
    """
    by_page: dict[tuple[str, int], list[float]] = defaultdict(list)
    for hit in hits:
        by_page[(hit.chunk.doc_id, hit.chunk.page)].append(hit.score)
    return {
        page: max(scores) + 0.1 * (sum(scores) - max(scores))
        for page, scores in by_page.items()
    }


def ranked_pages(hits: Sequence[Hit]) -> list[tuple[tuple[str, int], float]]:
    """Returns `(doc_id, page), score` pairs, best first."""
    return sorted(page_scores(hits).items(), key=lambda item: -item[1])
