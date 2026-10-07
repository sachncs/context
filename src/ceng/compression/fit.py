"""Hard-truncation fitting shared by the overflow policy and Truncate."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ceng import messages as messages_lib
from ceng.tokenizers import base as tokenizer_base

MARKER = "\n[... truncated ...]\n"
MAX_PASSES = 4


def total_tokens(
    messages: Sequence[messages_lib.Message],
    tokenizer: tokenizer_base.Tokenizer,
) -> int:
    """Returns the summed content tokens of `messages`."""
    return sum(tokenizer.count(m.content) for m in messages)


def largest_index(
    messages: Sequence[messages_lib.Message],
    tokenizer: tokenizer_base.Tokenizer,
) -> int:
    """Returns the index of the biggest message, preferring non-system."""
    candidates = [
        i for i, m in enumerate(messages) if m.role != messages_lib.Role.SYSTEM
    ] or list(range(len(messages)))
    return max(candidates, key=lambda i: tokenizer.count(messages[i].content))


def cut(
    content: str,
    keep_tokens: int,
    side: str,
    tokenizer: tokenizer_base.Tokenizer,
) -> str:
    """Keeps about `keep_tokens` of `content` and marks the removal.

    Args:
        content: Text to shorten.
        keep_tokens: Tokens of original content to retain.
        side: "head" keeps the start, "tail" the end, "middle" both ends.
        tokenizer: Counter used for sizing.
    """
    marker_cost = tokenizer.count(MARKER)
    keep = max(0, keep_tokens - marker_cost)
    if side == "head":
        return tokenizer.truncate(content, keep) + MARKER
    if side == "tail":
        kept = tail_of(content, keep, tokenizer)
        return MARKER + kept
    first = tokenizer.truncate(content, keep - keep // 2)
    last = tail_of(content, keep // 2, tokenizer)
    return first + MARKER + last


def tail_of(
    content: str, keep_tokens: int, tokenizer: tokenizer_base.Tokenizer
) -> str:
    """Returns the longest suffix of `content` within `keep_tokens`."""
    if keep_tokens <= 0:
        return ""
    low, high = 0, len(content)
    while low < high:
        mid = (low + high) // 2
        if tokenizer.count(content[mid:]) <= keep_tokens:
            high = mid
        else:
            low = mid + 1
    return content[low:]


def fit(
    messages: tuple[messages_lib.Message, ...],
    budget_tokens: int,
    tokenizer: tokenizer_base.Tokenizer,
    side: str = "tail",
) -> tuple[messages_lib.Message, ...]:
    """Shrinks the largest messages until the total fits `budget_tokens`.

    Args:
        messages: Messages to fit.
        budget_tokens: Ceiling for the summed content tokens.
        tokenizer: Counter.
        side: Which part of an oversize message survives; see `cut`.

    Returns:
        Messages within budget, or the best effort after `MAX_PASSES`
        rounds per message if the budget is unreachable (callers re-check).
    """
    current = messages
    passes_left = MAX_PASSES * max(1, len(messages))
    while passes_left > 0:
        passes_left -= 1
        excess = total_tokens(current, tokenizer) - budget_tokens
        if excess <= 0:
            break
        index = largest_index(current, tokenizer)
        content = current[index].content
        size = tokenizer.count(content)
        shorter = cut(content, size - excess, side, tokenizer)
        if shorter == content:
            break
        updated = list(current)
        updated[index] = dataclasses.replace(current[index], content=shorter)
        current = tuple(updated)
    return current
