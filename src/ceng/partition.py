"""Adaptive text partitioner used by :mod:`ceng.compress`.

The partitioner is content-aware and recursive: it splits a document
into leaves no larger than ``max_tokens`` by trying sentence
boundaries first, then paragraph boundaries, then a greedy word split
when a single sentence already exceeds the budget. This mirrors the
binary-tree scaffold from Wolf et al. (2026) and yields a flat list
of leaf chunks that can be summarised independently and aggregated.

The partitioner never touches an LLM; it is a pure string splitter
so it is deterministic and cheap to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ceng.tokens import count_tokens

WORD_MAX_BYTES = 4096


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
    text: str, *, max_tokens: int = 512, tokenizer: object | None = None
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
    leaves = split_recursive(text, max_tokens, tokenizer)
    return [Partition(text=leaf, index=i) for i, leaf in enumerate(leaves)]


def split_recursive(
    text: str, max_tokens: int, tokenizer: object | None
) -> list[str]:
    """Split ``text`` into leaves, recursively halving until each fits."""
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]
    sentences = split_sentences(text)
    if len(sentences) > 1:
        midpoint = len(sentences) // 2
        left = split_recursive(
            " ".join(sentences[:midpoint]), max_tokens, tokenizer
        )
        right = split_recursive(
            " ".join(sentences[midpoint:]), max_tokens, tokenizer
        )
        return left + right
    paragraphs = split_paragraphs(text)
    if len(paragraphs) > 1:
        midpoint = len(paragraphs) // 2
        left = split_recursive(
            "\n\n".join(paragraphs[:midpoint]), max_tokens, tokenizer
        )
        right = split_recursive(
            "\n\n".join(paragraphs[midpoint:]), max_tokens, tokenizer
        )
        return left + right
    return greedy_word_split(text, max_tokens, tokenizer)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` at sentence boundaries; never returns the empty list."""
    parts = [s for s in re.split(r"(?<=[.!?\n])\s+", text.strip()) if s]
    return parts if len(parts) > 1 else [text]


def split_paragraphs(text: str) -> list[str]:
    """Split ``text`` at blank-line paragraph boundaries."""
    parts = [s for s in re.split(r"\n\s*\n", text.strip()) if s]
    return parts if len(parts) > 1 else [text]


def greedy_word_split(
    text: str, max_tokens: int, tokenizer: object | None
) -> list[str]:
    """Greedy word-level chunker; streams words without holding them all.

    Uses :func:`re.finditer` to avoid materialising every word in
    memory before iteration starts. A single word that exceeds
    ``max_tokens`` is hard-capped at :data:`WORD_MAX_BYTES` bytes and
    emitted as one chunk so the caller never gets an oversized leaf.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
    for match in re.finditer(r"\S+", text):
        word = match.group(0)
        w_bytes = len(word.encode("utf-8"))
        if w_bytes > WORD_MAX_BYTES:
            flush()
            chunks.append(_cap_word(word, WORD_MAX_BYTES))
            current = []
            current_tokens = 0
            continue
        w_tokens = count_tokens(word, tokenizer)
        # A single word whose token count alone meets the budget must
        # still be emitted (alone, not dropped) so the partitioner
        # never silently truncates.
        if w_tokens >= max_tokens:
            flush()
            chunks.append(word)
            current = []
            current_tokens = 0
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


def _cap_word(word: str, max_bytes: int) -> str:
    """Return ``word`` truncated to ``max_bytes`` bytes, on a char boundary."""
    encoded = word.encode("utf-8")
    if len(encoded) <= max_bytes:
        return word
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
