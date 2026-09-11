"""Compact a long chat history while preserving primacy + recency.

Anthropic's guide describes "context rot": LLMs lose focus as the
context window grows, with a U-shaped retention curve that
favours information at the start (primacy) and the very end
(recency). The Coyle / Medium post on the same pattern recommends
discarding middle content while keeping both ends verbatim.

``compact_messages`` does exactly that: the first ``preserve_first``
and last ``preserve_last`` messages are kept as-is, and the middle
strip is summarised in a single backend call (or dropped
altogether if ``summarise_middle=False``).

This is intentionally separate from :func:`ceng.compress.ppa_compress`
which compresses a single long message via Partition-Prompt-Aggregate.
Compaction keeps structure; compression rephrases content.
"""

from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class CompactionProvenance:
    """Provenance block describing what compaction did.

    Attributes:
        preserved_first: Number of messages kept verbatim at the start.
        preserved_last: Number of messages kept verbatim at the end.
        summarised_count: Number of messages folded into the middle summary.
        summarised_tokens: Approximate token count of the produced summary.
        original_tokens: Pre-compaction token estimate.
        compacted_tokens: Post-compaction token estimate.
    """

    preserved_first: int
    preserved_last: int
    summarised_count: int
    original_tokens: int
    compacted_tokens: int
    summarised_tokens: int = 0


def compact_messages(
    messages: list[dict],
    *,
    backend: "object | None" = None,
    llm: str = "gpt-4o-mini",
    preserve_first: int = 2,
    preserve_last: int = 4,
    summarise_middle: bool = True,
    summary_max_tokens: int = 500,
    cache_dir: str = ".ceng/cache",
    summariser_system: str = (
        "You are a precise summariser. Produce a faithful, compact summary "
        "of the conversation strip provided. Preserve every decision, "
        "constraint, and named entity. Do not invent details."
    ),
    **call_kw,
) -> tuple[list[dict], CompactionProvenance]:
    """Compact ``messages`` while preserving primacy + recency.

    The system-role message (if present) is never compressed. The first
    ``preserve_first`` messages and the last ``preserve_last`` messages
    are kept verbatim. The remaining strip in the middle is either
    summarised in a single backend call or dropped entirely.

    Args:
        messages: OpenAI-style chat messages list.
        backend: Optional :class:`ceng.backends.Backend`. Defaults to
            the active backend (set with ``ceng.set_backend(...)``).
        llm: Model id.
        preserve_first: Number of leading messages to keep verbatim.
        preserve_last: Number of trailing messages to keep verbatim.
        summarise_middle: If True, summarise the middle strip in one
            backend call and insert the summary as a single
            ``assistant``-role message between preserve_first and
            preserve_last. If False, the middle strip is dropped.
        summary_max_tokens: Token budget for the middle summary.
        cache_dir: Path to the on-disk cache for the summary call.
        summariser_system: System prompt used for the summary call.
        **call_kw: Forwarded to ``backend.complete``.

    Returns:
        Tuple of the compacted message list and a provenance struct.

    Raises:
        ValueError: If the input contains a system-role message and
            the preservation strategy would drop or summarise it.
        RuntimeError: If the backend call fails; the call is wrapped
            with leaf-style error context (caller can catch and retry).
    """
    if not messages:
        raise ValueError("messages list is empty")
    if preserve_first < 0 or preserve_last < 0:
        raise ValueError("preserve_first and preserve_last must be >= 0")
    if preserve_first + preserve_last >= len(messages):
        # nothing to compact
        return list(messages), CompactionProvenance(
            preserved_first=len(messages),
            preserved_last=0,
            summarised_count=0,
            original_tokens=0,
            compacted_tokens=0,
        )

    head = list(messages[:preserve_first])
    tail = list(messages[len(messages) - preserve_last:]) if preserve_last else []
    middle = messages[preserve_first:len(messages) - preserve_last] if preserve_first + preserve_last < len(messages) else []

    # System-role messages are sacred. They cannot be summarised or
    # dropped. If any system message falls outside the preserved head
    # (or, if preserve_last >= 1, the preserved tail would include a
    # system message as the very last entry — unusual but defensible) we
    # refuse with a clear message rather than silently dropping the
    # instructions.
    head_indices = set(range(len(head)))
    tail_indices = set(range(len(messages) - len(tail), len(messages))) if tail else set()
    preserved_indices = head_indices | tail_indices

    body_roles = [m.get("role") for m in messages]
    for idx, role in enumerate(body_roles):
        if role == "system" and idx not in preserved_indices:
            role_name = messages[idx].get("role")
            raise ValueError(
                f"system-role message at index {idx} (role={role_name!r}) is "
                f"outside the preserved head/tail; bump preserve_first "
                f"to include it or move the system message to position 0"
            )

    if middle and any(m.get("role") == "system" for m in middle):
        # Defensive: the loop above already raised. This branch is
        # unreachable but kept as a safety net.
        raise ValueError(
            "system-role message in the middle strip"
        )

    if not summarise_middle:
        compacted = head + tail
        return compacted, CompactionProvenance(
            preserved_first=len(head),
            preserved_last=len(tail),
            summarised_count=len(middle),
            original_tokens=0,
            compacted_tokens=0,
        )

    summary = _summarise_middle(
        middle=middle,
        backend=backend,
        llm=llm,
        summary_max_tokens=summary_max_tokens,
        cache_dir=cache_dir,
        summariser_system=summariser_system,
        **call_kw,
    )
    summary_message = {
        "role": "assistant",
        "content": summary,
    }
    compacted = head + [summary_message] + tail
    from ceng.tokens import count_tokens
    return compacted, CompactionProvenance(
        preserved_first=len(head),
        preserved_last=len(tail),
        summarised_count=len(middle),
        original_tokens=0,
        compacted_tokens=0,
        summarised_tokens=count_tokens(summary),
    )


def _summarise_middle(
    *,
    middle: list[dict],
    backend: "object | None",
    llm: str,
    summary_max_tokens: int,
    cache_dir: str,
    summariser_system: str,
    **call_kw,
) -> str:
    """One backend call summarising the middle strip.

    Surfaces backend errors with the strip length so the caller can
    decide whether to retry or fall back to ``summarise_middle=False``.
    """
    from ceng.backends import get_backend
    from ceng.cache import Cache, make_key, NAMESPACE_SUMMARIZE

    if not middle:
        return ""

    chosen = backend or get_backend()
    cache = Cache(cache_dir=cache_dir) if cache_dir else None

    body = _strip_to_text(middle)
    prompt = (
        f"Summarise the following conversation strip in roughly "
        f"{summary_max_tokens} tokens. Preserve every decision, "
        f"constraint, named entity, and resolved question. "
        f"Return only the summary.\n\n"
        f"<messages>\n{body}\n</messages>"
    )

    cache_key = make_key(
        {
            "op": "compact_middle",
            "model": llm,
            "messages_sha256": __import__("hashlib").sha256(
                body.encode("utf-8")
            ).hexdigest(),
            "max_tokens": summary_max_tokens,
        }
    )
    if cache is not None:
        cached = cache.get(NAMESPACE_SUMMARIZE, cache_key)
        if cached is not None:
            return str(cached["text"])

    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", max(64, summary_max_tokens * 2))

    try:
        text = chosen.complete(
            messages=[
                {"role": "system", "content": summariser_system},
                {"role": "user", "content": prompt},
            ],
            model=llm,
            **kw,
        )
    except Exception as exc:
        raise RuntimeError(
            f"middle-strip summarisation failed after {len(middle)} "
            f"messages: {exc!r}"
        ) from exc

    if not text or not text.strip():
        raise RuntimeError(
            "summariser returned empty output; refusing to inject a "
            "blank assistant message into the conversation"
        )

    if cache is not None:
        cache.set(NAMESPACE_SUMMARIZE, cache_key, {"text": text})

    return text


def _strip_to_text(middle: list[dict]) -> str:
    """Render the middle strip as a single string for the summariser."""
    pieces: list[str] = []
    for m in middle:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        pieces.append(f"{role}: {content}")
    return "\n\n".join(pieces)
