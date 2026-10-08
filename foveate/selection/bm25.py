"""BM25 lexical retrieval (pure Python, no dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from foveate import errors
from foveate.documents import chunking
from foveate.selection import base

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-cases and splits text into index terms (stopwords removed)."""
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


@base.Retriever.registry.register("bm25")
class BM25Retriever(base.Retriever):
    """Okapi BM25 over chunk text and heading.

    Attributes:
        k1: Term-frequency saturation.
        b: Length normalisation strength.
    """

    def __init__(
        self,
        chunks: Sequence[chunking.Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Builds the inverted index.

        Raises:
            ConfigError: For non-positive `k1` or `b` outside [0, 1].
        """
        if k1 <= 0 or not 0 <= b <= 1:
            raise errors.ConfigError("need k1 > 0 and 0 <= b <= 1")
        self.chunks = tuple(chunks)
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for index, chunk in enumerate(self.chunks):
            terms = tokenize(chunk.searchable)
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((index, count))
        total = sum(self.lengths)
        self.average = total / len(self.lengths) if self.lengths else 0.0

    def idf(self, term: str) -> float:
        """Returns the (non-negative) inverse document frequency of `term`."""
        n = len(self.chunks)
        df = len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    async def search(self, query: str, k: int) -> list[base.Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            weight = self.idf(term)
            for index, tf in self.postings.get(term, ()):
                norm = (
                    1
                    - self.b
                    + self.b * self.lengths[index] / (self.average or 1)
                )
                scores[index] += (
                    weight * tf * (self.k1 + 1) / (tf + self.k1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [base.Hit(self.chunks[i], s) for i, s in ranked[:k]]
