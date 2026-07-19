"""Cheap token counting used to budget long contexts.

When ``tiktoken`` is installed (``pip install ceng[tokenize]``) we use
it for accurate counts; otherwise we fall back to a heuristic
``len(text) // 4``. Heuristic counts are good enough for budgeting
chunks and never make it into the prompt sent to the model.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

_TOKENIZER = None


def _get_tokenizer():
    """Return a cached ``tiktoken`` encoding, or ``None`` if not installed."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    if importlib.util.find_spec("tiktoken") is None:
        _TOKENIZER = False
        return None
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str, tokenizer: Optional[object] = None) -> int:
    """Return the number of tokens in ``text``.

    Args:
        text: Input string. Must be a ``str``.
        tokenizer: Optional pre-built ``tiktoken`` encoding. If omitted,
            :func:`_get_tokenizer` is consulted and falls back to the
            heuristic when ``tiktoken`` is unavailable.

    Returns:
        Estimated token count. Always at least one for non-empty input.
    """
    if not text:
        return 0
    enc = tokenizer if tokenizer is not None else _get_tokenizer()
    if enc is None or enc is False:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def token_budget_split(
    text: str, max_tokens: int, tokenizer: Optional[object] = None
) -> list[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` tokens.

    Uses sentence boundaries first, falling back to a greedy word split
    when a sentence is itself over budget.

    Args:
        text: Input text to split.
        max_tokens: Hard ceiling for any single chunk.
        tokenizer: Optional ``tiktoken`` encoding; same resolution as
            :func:`count_tokens`.

    Returns:
        A list of non-empty text chunks. A single oversized sentence is
        returned as one chunk rather than truncated.
    """
    import re

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text:
        return []
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    sentences = [s for s in re.split(r"(?<=[.!?\n])\s+", text) if s]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        s_tokens = count_tokens(sentence, tokenizer)
        if s_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(_greedy_word_split(sentence, max_tokens, tokenizer))
            continue
        if current_tokens + s_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _greedy_word_split(
    text: str, max_tokens: int, tokenizer: Optional[object]
) -> list[str]:
    """Greedy word-level split of a sentence that exceeds ``max_tokens``."""
    import re

    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        w_tokens = count_tokens(word, tokenizer)
        if current_tokens + w_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += w_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
