"""Adaptive text partitioner used by :mod:`ceng.compress`.

The partitioner is content-aware and recursive: it splits a document
into leaves no larger than ``max_tokens`` by trying sentence
boundaries first, then paragraph boundaries, then a greedy word split
when a single sentence already exceeds the budget. This mirrors the
binary-tree scaffold from Wolf et al. (2026) and yields a flat list of
leaf chunks that can be summarised independently and aggregated.

Ponytail: the partitioner never knows about LLMs. It is a pure string
splitter, so it is deterministic and cheap to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ceng.tokens import count_tokens


@dataclass(frozen=True)
class Partition:
    """A leaf chunk of a partitioned document.

    Attributes:
        text: The chunk's text content.
        index: Zero-based ordinal within the original document, in
            document order.
    """

    text: str
    index: int


def partition_text(
    text: str, *, max_tokens: int = 512, tokenizer: Optional[object] = None
) -> list[Partition]:
    """Return the leaf chunks of an adaptive partition of ``text``.

    Chunks are non-empty, ordered by their position in ``text``, and
    each fits inside ``max_tokens`` tokens (using the same counter the
    caller passed, or the heuristic default from :mod:`ceng.tokens`).

    Args:
        text: Source text to partition.
        max_tokens: Hard ceiling for any single chunk. Must be > 0.
        tokenizer: Optional ``tiktoken`` encoding for accurate counts.

    Returns:
        List of :class:`Partition` leaves in document order.

    Raises:
        ValueError: If ``max_tokens`` is not positive or ``text`` is empty.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text:
        raise ValueError("text must not be empty")
    leaves = _split_recursive(text, max_tokens, tokenizer)
    return [Partition(text=leaf, index=i) for i, leaf in enumerate(leaves)]


def _split_recursive(
    text: str, max_tokens: int, tokenizer: Optional[object]
) -> list[str]:
    """Split ``text`` into leaves, recursively halving until each fits.

    A single sentence that already exceeds ``max_tokens`` is emitted
    as-is rather than being silently truncated.
    """
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        midpoint = len(sentences) // 2
        left = _split_recursive(" ".join(sentences[:midpoint]), max_tokens, tokenizer)
        right = _split_recursive(" ".join(sentences[midpoint:]), max_tokens, tokenizer)
        return left + right
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        midpoint = len(paragraphs) // 2
        left = _split_recursive("\n\n".join(paragraphs[:midpoint]), max_tokens, tokenizer)
        right = _split_recursive("\n\n".join(paragraphs[midpoint:]), max_tokens, tokenizer)
        return left + right
    return _greedy_word_split(text, max_tokens, tokenizer)


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` at sentence boundaries; never returns the empty list."""
    parts = [s for s in re.split(r"(?<=[.!?\n])\s+", text.strip()) if s]
    return parts if len(parts) > 1 else [text]


def _split_paragraphs(text: str) -> list[str]:
    """Split ``text`` at blank-line paragraph boundaries."""
    parts = [s for s in re.split(r"\n\s*\n", text.strip()) if s]
    return parts if len(parts) > 1 else [text]


def _greedy_word_split(
    text: str, max_tokens: int, tokenizer: Optional[object]
) -> list[str]:
    """Greedy word-level chunker; chunk never exceeds ``max_tokens``."""
    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        w_tokens = count_tokens(word, tokenizer)
        # If a single word is over budget, emit it on its own rather
        # than silently dropping it. The aggregate summary will carry
        # it verbatim.
        if w_tokens >= max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.append(word)
            continue
        if current_tokens + w_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += w_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
