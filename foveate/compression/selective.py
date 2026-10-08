"""LLM-free pruning by information content, at phrase granularity.

Selective Context (Li et al., 2023) removes the parts of a prompt that a
language model finds predictable, and finds phrases a better unit than tokens
or sentences. This version needs no model: a phrase's information is the mean
surprisal of its words under word frequencies counted in the text itself, with
a bonus for numbers and capitalised names, which carry the facts. The
lowest-information phrases are dropped until the text fits.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import re
from typing import TYPE_CHECKING

from foveate import errors
from foveate.compression import base, fit, report
from foveate.tokenizers import base as tokenizer_base

if TYPE_CHECKING:
    from foveate import context as context_lib

PHRASE_BOUNDARY = re.compile(r"(?<=[.!?;:,])\s+|\n+|\s+[-(]\s*")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.,%$-]*")


@base.Compressor.register("selective")
@dataclasses.dataclass(frozen=True)
class Selective(base.Compressor):
    """Keeps the highest-information phrases of each message, in order.

    Attributes:
        fact_bonus: Multiplier on the score of phrases that contain numbers
            or capitalised names.
        min_phrase_tokens: Phrases shorter than this are merged into the
            next one so units stay meaningful.
    """

    fact_bonus: float = 1.5
    min_phrase_tokens: int = 3

    def __post_init__(self) -> None:
        if self.fact_bonus < 1.0 or self.min_phrase_tokens < 1:
            raise errors.ConfigError(
                "fact_bonus must be >= 1 and min_phrase_tokens >= 1"
            )

    def phrases(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        """Splits text into phrases, merging very short ones forward."""
        out: list[str] = []
        pending = ""
        for piece in PHRASE_BOUNDARY.split(text):
            piece = piece.strip()
            if not piece:
                continue
            pending = f"{pending} {piece}".strip()
            if tokenizer.count(pending) >= self.min_phrase_tokens:
                out.append(pending)
                pending = ""
        if pending:
            out.append(pending)
        return out

    def scores(self, phrases: list[str]) -> list[float]:
        """Returns one information score per phrase."""
        words = [[w.casefold() for w in WORD.findall(p)] for p in phrases]
        counts = collections.Counter(w for ws in words for w in ws)
        total = sum(counts.values()) + len(counts)
        scores = []
        for phrase, ws in zip(phrases, words, strict=True):
            if not ws:
                scores.append(0.0)
                continue
            surprisal = sum(-math.log((counts[w] + 1) / total) for w in ws)
            score = surprisal / len(ws)
            has_fact = any(c.isdigit() for c in phrase) or any(
                w[:1].isupper() for w in phrase.split()[1:]
            )
            scores.append(score * (self.fact_bonus if has_fact else 1.0))
        return scores

    def prune(
        self,
        text: str,
        target: int,
        tokenizer: tokenizer_base.Tokenizer,
    ) -> str:
        """Returns the best phrases of `text` within `target` tokens."""
        phrases = self.phrases(text, tokenizer)
        if len(phrases) < 2:
            return text
        scores = self.scores(phrases)
        ranked = sorted(range(len(phrases)), key=lambda i: -scores[i])
        chosen: set[int] = set()
        used = 0
        for index in ranked:
            cost = tokenizer.count(phrases[index]) + 1
            if used + cost <= target:
                chosen.add(index)
                used += cost
        return " ".join(phrases[i] for i in sorted(chosen))

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
            text = self.prune(original, target, tokenizer)
            trace.add(
                report.StepRecord(
                    name=f"m{index} prune",
                    input_tokens=tokenizer.count(original),
                    output_tokens=tokenizer.count(text),
                )
            )
            messages[index] = messages[index].with_content(text)
        return dataclasses.replace(context, messages=tuple(messages))
