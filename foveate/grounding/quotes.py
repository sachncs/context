"""Deterministic checking that a quote really appears on a page."""

from __future__ import annotations

import re

NUMBER_COMMAS = re.compile(r"(?<=\d),(?=\d{3}\b)")
NON_WORD = re.compile(r"[^0-9a-z.%]+")
SHORT_QUOTE_WORDS = 3
THRESHOLD = 0.85


def normalise(text: str) -> str:
    """Lower-cases and strips formatting so equal content compares equal.

    Thousands separators, currency signs, punctuation and whitespace runs
    are dropped ("$1,577.00" and "1577.00" match).
    """
    folded = NUMBER_COMMAS.sub("", text.casefold().replace("$", " "))
    words = (w.strip(".") for w in NON_WORD.sub(" ", folded).split())
    return " ".join(w for w in words if w)


def lcs_length(needle: list[str], haystack: list[str]) -> int:
    """Returns the longest-common-subsequence length of two word lists."""
    previous = [0] * (len(haystack) + 1)
    for word in needle:
        current = [0]
        for index, other in enumerate(haystack, start=1):
            current.append(
                previous[index - 1] + 1
                if word == other
                else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1]


def supports(quote: str, page_text: str, threshold: float = THRESHOLD) -> bool:
    """Returns whether `quote` is (nearly) a verbatim excerpt of `page_text`.

    Short quotes (under three words, typically a figure) must match exactly
    after normalisation. Longer quotes pass if at least `threshold` of their
    words appear in order, which tolerates model re-spacing and dropped
    punctuation but not invented content.
    """
    wanted = normalise(quote)
    if not wanted:
        return False
    page = normalise(page_text)
    if wanted in page:
        return True
    words = wanted.split()
    if len(words) < SHORT_QUOTE_WORDS:
        return False
    return lcs_length(words, page.split()) / len(words) >= threshold
