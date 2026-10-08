"""Prompt-injection hardening for untrusted text placed inside prompts.

Retrieved pages are data, not instructions. `escape` stops that data from
closing the fence that surrounds it or from forging a page marker, and
`suspicious` flags the common phrasing of instruction-hijacking attempts so
callers can log, drop or review them. Neither is a complete defence; they
remove the cheap attacks and the prompt tells the model to treat the fenced
block as data.
"""

from __future__ import annotations

import re

ZWSP = "\u200b"
FENCE_TAGS = ("pages", "outline", "text", "data", "context", "document")
FENCE = re.compile(r"<(/?)\s*(" + "|".join(FENCE_TAGS) + r")\b", re.IGNORECASE)
MARKER = re.compile(r"^(\s*)\[([^\]\n]{1,80}?\sp\.\d+[^\]\n]*)\]", re.MULTILINE)
PATTERNS = (
    re.compile(
        r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b"
        r"(previous|prior|above|earlier|all)\b[^.\n]{0,40}"
        r"\b(instructions?|prompts?|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(you are now|act as|pretend to be)\b", re.IGNORECASE),
    re.compile(r"\b(system prompt|developer message)\b", re.IGNORECASE),
    re.compile(
        r"\b(reveal|print|output)\b[^.\n]{0,30}\b(secret|password|api key)\b",
        re.IGNORECASE,
    ),
    re.compile(r"</?\s*(system|assistant|im_start|im_end)\b", re.IGNORECASE),
)


def escape(text: str) -> str:
    """Neutralises fence tags and forged page markers in untrusted text.

    A zero-width space is inserted after the `<` of a fence tag, so it still
    reads the same to a person; a line that looks like a page marker gets a
    backslash prefix.
    """
    text = FENCE.sub(lambda m: f"<{ZWSP}{m.group(1)}{m.group(2)}", text)
    return MARKER.sub(lambda m: f"{m.group(1)}\\[{m.group(2)}]", text)


def suspicious(text: str) -> list[str]:
    """Returns the snippets of `text` that look like injected instructions."""
    found = []
    for pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(match.group(0))
    return found
