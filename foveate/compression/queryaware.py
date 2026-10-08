"""LLM-free compression that keeps what a question asks about.

`Extractive` keeps sentences made of frequent words. `QueryExtractive` keeps
the sentences that share rare words with the query (BM25-style weights), which
is what a condensed neighbour page should do: preserve the sentence that may
hold the answer, not the most generic one.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import TYPE_CHECKING

from foveate import errors
from foveate import messages as messages_lib
from foveate.compression import base, extractive, report

if TYPE_CHECKING:
    from foveate import context as context_lib

K1 = 1.2
TIE_BREAK = 1e-3


@base.Compressor.register("query")
@dataclasses.dataclass(frozen=True)
class QueryExtractive(extractive.Extractive):
    """Keeps the sentences most relevant to a query, in original order.

    Attributes:
        query: What the reader wants to know. When empty, `run` uses the last
            user message and `extract` falls back to frequency scoring.
    """

    query: str = ""

    def terms(self, text: str) -> list[str]:
        """Returns lower-cased content words of `text`."""
        return [
            w
            for w in extractive.WORD.findall(text.lower())
            if w not in extractive.STOPWORDS
        ]

    def score(self, sentences: list[str]) -> list[float]:
        wanted = set(self.terms(self.query))
        if not wanted:
            return super().score(sentences)
        words = [self.terms(s) for s in sentences]
        count = len(sentences)
        document_frequency: collections.Counter[str] = collections.Counter()
        for ws in words:
            document_frequency.update(set(ws) & wanted)
        fallback = super().score(sentences)
        top = max(fallback) or 1.0
        scores = []
        for ws, base_score in zip(words, fallback, strict=True):
            frequency = collections.Counter(ws)
            total = 0.0
            for term in wanted:
                if term not in frequency:
                    continue
                idf = math.log(
                    1.0
                    + (count - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                tf = frequency[term]
                total += idf * tf * (K1 + 1) / (tf + K1)
            scores.append(total + TIE_BREAK * base_score / top)
        return scores

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        if self.query:
            return await super().run(context, budget, trace)
        users = [
            m.content
            for m in context.messages
            if m.role is messages_lib.Role.USER
        ]
        if not users:
            raise errors.ConfigError(
                "query compression needs `query=` or a user message"
            )
        bound = dataclasses.replace(self, query=users[-1])
        return await bound.run(context, budget, trace)
