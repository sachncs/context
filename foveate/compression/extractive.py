"""LLM-free extractive compression (frequency-based sentence ranking)."""

from __future__ import annotations

import collections
import dataclasses
import re
from typing import TYPE_CHECKING

from foveate import errors
from foveate.compression import base, fit, report
from foveate.partition import base as partition_base
from foveate.tokenizers import base as tokenizer_base

if TYPE_CHECKING:
    from foveate import context as context_lib

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
        unit_tokens: Sentences longer than this are subdivided.
    """

    edge_bonus: float = 0.25
    unit_tokens: int = 64

    def __post_init__(self) -> None:
        if self.edge_bonus < 0:
            raise errors.ConfigError("edge_bonus must be >= 0")
        if self.unit_tokens < 1:
            raise errors.ConfigError("unit_tokens must be >= 1")

    def units(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        """Splits text into sentences, subdividing any oversize sentence.

        Text without sentence punctuation (logs, tables, word lists) is cut
        by the recursive partitioner so it can still be reduced.
        """
        splitter = partition_base.RecursivePartitioner(self.unit_tokens)
        units: list[str] = []
        for sentence in SENTENCE_BOUNDARY.split(text):
            stripped = sentence.strip()
            if not stripped:
                continue
            if tokenizer.count(stripped) > self.unit_tokens:
                units.extend(
                    part.text.strip()
                    for part in splitter.split(stripped, tokenizer)
                    if part.text.strip()
                )
            else:
                units.append(stripped)
        return units

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
        targets = fit.fair_targets(context.messages, budget.tokens, tokenizer)
        messages = list(context.messages)
        for index, target in targets.items():
            original = messages[index].content
            text = self.extract(original, target, tokenizer)
            trace.add(
                report.StepRecord(
                    name=f"m{index} extract",
                    input_tokens=tokenizer.count(original),
                    output_tokens=tokenizer.count(text),
                )
            )
            messages[index] = messages[index].with_content(text)
        return dataclasses.replace(context, messages=tuple(messages))

    def extract(
        self,
        text: str,
        target: int,
        tokenizer: tokenizer_base.Tokenizer,
    ) -> str:
        """Returns the best sentences of `text` within `target` tokens."""
        sentences = self.units(text, tokenizer)
        if len(sentences) < 2:
            return text
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
        return " ".join(sentences[i] for i in sorted(chosen))
