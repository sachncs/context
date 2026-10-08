"""Removing inline reasoning from model output."""

from __future__ import annotations

import re

CLOSED = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE
)
OPEN = re.compile(
    r"<(think|thinking|reasoning)>.*\Z", re.DOTALL | re.IGNORECASE
)
STRAY_CLOSE = re.compile(r"\A.*?</(think|thinking|reasoning)>\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Drops `<think>...</think>` style blocks that some models inline.

    An unterminated block (the model ran out of tokens while thinking) is
    dropped too, leaving only text that is safe to parse or show. A lone
    closing tag means the template opened the block, so everything before it
    is reasoning.
    """
    text = CLOSED.sub("", text)
    text = OPEN.sub("", text)
    if "<" in text:
        text = STRAY_CLOSE.sub("", text, count=1)
    return text.strip()
