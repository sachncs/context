"""Grounded answers: text plus verifiable evidence."""

from __future__ import annotations

import dataclasses

from foveate import usage as usage_lib
from foveate.foveation import Foveation

NOT_FOUND = "I could not find this in the provided documents."


@dataclasses.dataclass(frozen=True, slots=True)
class Citation:
    """A claim's evidence.

    Attributes:
        doc_id: Document the quote is from.
        page: Page number the model cited.
        quote: Excerpt the model copied from that page.
        verified: Whether the quote was found on that page of the source.
    """

    doc_id: str
    page: int
    quote: str
    verified: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Answer:
    """The result of `Foveator.ask`.

    Attributes:
        question: The question asked.
        text: The answer, or `NOT_FOUND` when abstaining.
        found: Whether the model found the answer in the documents.
        grounded: Whether it has at least one citation and every citation
            is verified against the cited page.
        abstained: Whether the answer is the not-found message.
        citations: Cited evidence, verified against the source pages.
        foveation: Which pages were shown at which fidelity (last round).
        rounds: Model rounds used (1 = answered on the first attempt).
        usage: Total provider token usage over all rounds.
        cost_usd: Estimated spend from the runtime's price table.
        seconds: Wall-clock duration.
        raw: The model's last raw reply (for debugging).
    """

    question: str
    text: str
    found: bool
    grounded: bool
    abstained: bool
    citations: tuple[Citation, ...]
    foveation: Foveation
    rounds: int
    usage: usage_lib.Usage
    cost_usd: float = 0.0
    seconds: float = 0.0
    raw: str = ""

    @property
    def pages(self) -> list[tuple[str, int]]:
        """Returns the distinct `(doc_id, page)` pairs that were cited."""
        return list(dict.fromkeys((c.doc_id, c.page) for c in self.citations))
