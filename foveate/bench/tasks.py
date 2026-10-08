"""Concrete benchmarks: FiNER, Formula, DDXPlus."""

from __future__ import annotations

import re
from collections.abc import Mapping

from foveate import errors
from foveate.bench import base
from foveate.evolution import grading

BIO_TAGS = (
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC",
    "O",
)


@base.Benchmark.register("finer")
class Finer(base.Benchmark):
    """Token tagging with BIO labels (exact sequence match).

    The packaged fixture uses the nine CoNLL-style tags below, not the 139
    XBRL types of the full FiNER-139 corpus; supply `tags` for real data.

    Attributes:
        tags: Allowed output labels.
    """

    cited_baseline = 70.7
    cited_ace = 78.3

    def __init__(self, tags: tuple[str, ...] = BIO_TAGS) -> None:
        self.tags = tags

    def row_to_sample(self, row: Mapping[str, object]) -> grading.Sample:
        tokens = row.get("tokens")
        labels = row.get("labels")
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise errors.ValidationError("needs 'tokens' and 'labels' lists")
        if len(tokens) != len(labels):
            raise errors.ValidationError("tokens and labels differ in length")
        sentence = str(row.get("sentence") or " ".join(map(str, tokens)))
        return grading.Sample(
            question=(
                f"Sentence: {sentence}\n"
                f"Tokens: {' '.join(map(str, tokens))}\n\n"
                "Tags (one per line):"
            ),
            target="\n".join(map(str, labels)),
            context=(
                "Tag each token with one of: "
                + ", ".join(self.tags)
                + ".\nReturn one tag per line, in token order."
            ),
            id=str(row.get("id") or ""),
        )

    @staticmethod
    def tag_sequence(text: str) -> tuple[str, ...]:
        """Returns the last whitespace-separated field of every line."""
        return tuple(
            line.split()[-1] for line in text.splitlines() if line.strip()
        )

    def extract_answer(self, text: str) -> str:
        return text.strip()

    def is_correct(self, predicted: str, target: str) -> bool:
        return self.tag_sequence(predicted) == self.tag_sequence(target)


NUMBER = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"
)


@base.Benchmark.register("formula")
class Formula(base.Benchmark):
    """Numeric financial questions, graded with a relative tolerance.

    Currency symbols and thousands separators are ignored; a trailing `%`
    is accepted either as the number itself or as a fraction.

    Attributes:
        tolerance: Allowed relative error.
    """

    cited_baseline = 67.5
    cited_ace = 85.5

    def __init__(self, tolerance: float = 0.01) -> None:
        if tolerance < 0:
            raise errors.ConfigError("tolerance must be non-negative")
        self.tolerance = tolerance

    def row_to_sample(self, row: Mapping[str, object]) -> grading.Sample:
        context = str(row.get("context") or row.get("passage") or "")
        question = base.required_text(row, "question", "query", "text")
        return grading.Sample(
            question=f"Context: {context}\n\nQuestion: {question}",
            target=base.required_text(row, "answer", "target"),
            context=context,
            id=str(row.get("id") or ""),
        )

    @staticmethod
    def candidates(text: str) -> list[float]:
        """Returns plausible numeric readings of the first number in text."""
        match = NUMBER.search(text.replace("$", ""))
        if match is None:
            return []
        value = float(match.group(0).replace(",", ""))
        rest = text.replace("$", "")[match.end() :].lstrip()
        return [value, value / 100.0] if rest.startswith("%") else [value]

    def is_correct(self, predicted: str, target: str) -> bool:
        wanted = self.candidates(target)
        if not wanted:
            return False
        for guess in self.candidates(predicted):
            for gold in wanted:
                if gold == 0:
                    if abs(guess) <= self.tolerance:
                        return True
                elif abs(guess - gold) / abs(gold) <= self.tolerance:
                    return True
        return False


INTEGER = re.compile(r"[-+]?\d+")


@base.Benchmark.register("ddxplus")
class DDXPlus(base.Benchmark):
    """Four-way differential diagnosis; the answer is an option index."""

    cited_baseline = 75.2
    cited_ace = 90.2

    def row_to_sample(self, row: Mapping[str, object]) -> grading.Sample:
        question = base.required_text(row, "question", "text")
        options = row.get("options") or row.get("choices")
        if not isinstance(options, list) or not options:
            raise errors.ValidationError("needs a non-empty 'options' list")
        listing = "\n".join(
            f"{i}. {option}" for i, option in enumerate(options)
        )
        return grading.Sample(
            question=(
                f"{question}\n\nOptions:\n{listing}\n\n"
                "Answer with the option number only."
            ),
            target=base.required_text(row, "answer", "label"),
            context=question,
            id=str(row.get("id") or ""),
        )

    @staticmethod
    def first_integer(text: str) -> int | None:
        """Returns the first integer in `text`, or None."""
        match = INTEGER.search(text)
        return int(match.group(0)) if match else None

    def is_correct(self, predicted: str, target: str) -> bool:
        guess = self.first_integer(predicted)
        return guess is not None and guess == self.first_integer(target)
