"""Haystacks of any length, and a place to hide a needle.

The synthetic prose is generated from a seed, so a haystack is reproducible
and needs no download. `book_pages` turns any real text (for example the
NoLiMa books) into pages the same way.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from foveate import errors
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.tokenizers import base as tokenizer_base

PAGE_TOKENS = 500
SUBJECTS = (
    "The committee",
    "A visiting engineer",
    "The regional office",
    "Our archivist",
    "The night shift",
    "A local supplier",
    "The auditor",
    "The planning group",
    "A junior analyst",
    "The harbour authority",
)
VERBS = (
    "reviewed",
    "postponed",
    "catalogued",
    "questioned",
    "summarised",
    "reorganised",
    "approved",
    "revisited",
    "measured",
    "documented",
)
OBJECTS = (
    "the quarterly timetable",
    "an old shipping ledger",
    "the parking plan",
    "a stack of survey forms",
    "the catering contract",
    "last year's budget",
    "the maintenance schedule",
    "a pile of travel receipts",
    "the visitor register",
    "the lighting upgrade",
)
ENDINGS = (
    "without reaching a decision.",
    "before the afternoon break.",
    "and filed a short note.",
    "while the weather cleared.",
    "and asked for more time.",
    "as usual.",
    "with no objections.",
)


def prose_sentences(count: int, rng: random.Random) -> list[str]:
    """Returns `count` bland, varied sentences that contain no facts of note."""
    return [
        f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} "
        f"{rng.choice(OBJECTS)} {rng.choice(ENDINGS)}"
        for place in range(count)
    ]


def prose_pages(
    tokens: int,
    tokenizer: tokenizer_base.Tokenizer,
    seed: int,
    page_tokens: int = PAGE_TOKENS,
) -> list[str]:
    """Builds pages of synthetic prose totalling at least `tokens` tokens."""
    if tokens < page_tokens:
        raise errors.ConfigError("tokens must be at least one page")
    rng = random.Random(seed)
    pages: list[str] = []
    total = 0
    while total < tokens:
        text = ""
        while tokenizer.count(text) < page_tokens:
            text += " ".join(prose_sentences(4, rng)) + "\n"
        pages.append(text.strip())
        total += tokenizer.count(text)
    return pages


def book_pages(
    text: str,
    tokens: int,
    tokenizer: tokenizer_base.Tokenizer,
    seed: int,
    page_tokens: int = PAGE_TOKENS,
) -> list[str]:
    """Cuts a window of `tokens` tokens out of `text` and pages it.

    The window starts at a seeded offset so different seeds read different
    parts of the book.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        raise errors.ValidationError("haystack text is empty")
    start = random.Random(seed).randrange(len(paragraphs))
    pages: list[str] = []
    current = ""
    total = 0
    index = start
    while total < tokens:
        current += paragraphs[index % len(paragraphs)] + "\n\n"
        index += 1
        if tokenizer.count(current) >= page_tokens:
            pages.append(current.strip())
            total += tokenizer.count(current)
            current = ""
        if index - start > 50 * len(paragraphs):
            raise errors.ValidationError("haystack text is too short")
    return pages


def insert(pages: Sequence[str], sentence: str, depth: float) -> list[str]:
    """Hides `sentence` at relative `depth` (0 = start, 1 = end).

    The sentence is placed on its own line in the middle of the page found at
    that depth, so it cannot be located by looking at page edges.
    """
    if not 0.0 <= depth <= 1.0:
        raise errors.ConfigError("depth must be between 0 and 1")
    out = list(pages)
    target = min(len(out) - 1, int(depth * len(out)))
    text = out[target]
    cut = text.find("\n", len(text) // 2)
    cut = len(text) if cut == -1 else cut
    out[target] = f"{text[:cut]}\n{sentence}\n{text[cut:]}".strip()
    return out


def page_of(count: int, depth: float) -> int:
    """Returns the 1-based page `insert` uses for `count` pages."""
    return min(count - 1, int(depth * count)) + 1


def to_document(
    doc_id: str,
    pages: Sequence[str],
    tokenizer: tokenizer_base.Tokenizer,
) -> Document:
    """Wraps page texts in a `Document`."""
    return Document(
        doc_id,
        tuple(
            page_lib.Page(i, text, tokenizer.count(text), ())
            for i, text in enumerate(pages, start=1)
        ),
    )
