"""LLM-free extractive compression (frequency-based sentence ranking)."""

from __future__ import annotations

import collections
import dataclasses
import re
from typing import TYPE_CHECKING

from ceng import errors
from ceng.compression import base, fit, report

if TYPE_CHECKING:
    from ceng import context as context_lib

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[A-Za-z0-9']+")
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
        "will",
        "with",
    ]
)


@base.Compressor.register("extractive")
@dataclasses.dataclass(frozen=True)
class Extractive(base.Compressor):
    """Keeps the highest-scoring sentences of the largest message.

    Sentences are scored by the average document frequency of their content
    words plus a small bonus for the first and last sentence. The kept
    sentences stay in original order. No network access is needed, which
    makes this the natural offline fallback for LLM strategies.

    Attributes:
        edge_bonus: Score multiplier added to the first and last sentence.
    """

    edge_bonus: float = 0.25

    def __post_init__(self) -> None:
        if self.edge_bonus < 0:
            raise errors.ConfigError("edge_bonus must be >= 0")

    def score(self, sentences: list[str]) -> list[float]:
        """Returns one relevance score per sentence."""
        words = [
            [w for w in WORD.findall(s.lower()) if w not in STOPWORDS]
            for s in sentences
        ]
        frequency = collections.Counter(w for ws in words for w in ws)
        scores = []
        for i, ws in enumerate(words):
            base_score = sum(frequency[w] for w in ws) / len(ws) if ws else 0.0
            if i in (0, len(sentences) - 1):
                base_score *= 1.0 + self.edge_bonus
            scores.append(base_score)
        return scores

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        tokenizer = context.runtime.tokenizer
        index = fit.largest_index(context.messages, tokenizer)
        message = context.messages[index]
        others = context.token_count - tokenizer.count(message.content)
        target = max(1, budget.tokens - others)
        sentences = [
            s.strip()
            for s in SENTENCE_BOUNDARY.split(message.content)
            if s.strip()
        ]
        if len(sentences) < 2:
            return context
        scores = self.score(sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])
        chosen: set[int] = set()
        used = 0
        for i in ranked:
            cost = tokenizer.count(sentences[i]) + 1
            if used + cost > target:
                continue
            chosen.add(i)
            used += cost
        text = " ".join(sentences[i] for i in sorted(chosen))
        trace.add(
            report.StepRecord(
                name=f"kept {len(chosen)}/{len(sentences)} sentences",
                input_tokens=tokenizer.count(message.content),
                output_tokens=tokenizer.count(text),
            )
        )
        messages = list(context.messages)
        messages[index] = message.with_content(text)
        return dataclasses.replace(context, messages=tuple(messages))
