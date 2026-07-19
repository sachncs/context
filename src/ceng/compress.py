"""Partition-Prompt-Aggregate compression for long contexts.

``ppa_compress`` takes a chat-message list, finds the largest user-role
message, partitions it into token-bounded leaves (via
:func:`ceng.partition.partition_text`), summarises each leaf in
isolation (cached), then aggregates the leaf summaries with one final
combiner call to produce a single coherent condensed message.

This is the macro-fallacy-resistant path: instead of asking one model
"please summarise this whole document" (which loses detail), we ask
the model about each chunk and then aggregate the chunk-level
summaries (which preserves more unique facts).

The two LLM calls per chunk are:

* **Leaf summary** — ``SUMMARIZE_PROMPT``. One call per leaf, cache key
  derived from ``(model, text_sha256, prompt_template_version)``.
* **Combine** — ``COMBINE_PROMPT``. One call on the list of leaf
  summaries.

If a message already fits the budget, the function short-circuits and
returns the input untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from ceng.backends import Backend, get_backend
from ceng.cache import NAMESPACE_SUMMARIZE, Cache, make_key
from ceng.partition import partition_text
from ceng.tokens import count_tokens


PROMPT_VERSION = "1"
SUMMARIZE_SYSTEM = (
    "You are a precise summariser. Preserve every unique fact, entity, "
    "and number. Do not invent details."
)
COMBINE_SYSTEM = (
    "You are a precise editor. Combine several section summaries of the "
    "same document into ONE coherent summary. Preserve every unique fact, "
    "entity, and number across sections. Do not invent details."
)


@dataclass
class CompressResult:
    """Stats for a :func:`ppa_compress` call.

    Attributes:
        messages: The compressed message list (also returned from the
            function).
        original_tokens: Token count of the source message.
        compressed_tokens: Token count of the compressed message. May be
            slightly over ``budget_tokens`` if the model returns more
            than asked; budget is a soft target.
        leaf_count: Number of leaf partitions produced.
        cache_hits: Number of leaf summaries served from cache.
        cache_misses: Number of leaf summaries that had to be computed.
        backend: The backend instance used.
        model: Model id used.
    """

    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    leaf_count: int
    cache_hits: int
    cache_misses: int
    backend: Backend
    model: str


def ppa_compress(
    messages: list[dict],
    *,
    budget_tokens: int,
    llm: str = "gpt-4o-mini",
    cache_dir: str = ".ceng_cache",
    backend: Optional[Backend] = None,
    tokenizer: Any = None,
    summary_max_tokens: Optional[int] = None,
    partition_max_tokens: Optional[int] = None,
    **call_kw: Any,
) -> list[dict]:
    """Compress the longest user message in ``messages``.

    Args:
        messages: OpenAI-style chat messages list. The longest message
            whose ``role == "user"`` is the compression target. If no
            such message exists, the list is returned unchanged.
        budget_tokens: Target token count for the compressed message.
            Soft target — the actual output is bounded by the model's
            response and the combiner prompt.
        llm: Model id accepted by the active backend.
        cache_dir: Path to the on-disk cache directory.
        backend: Optional :class:`Backend` instance. Defaults to the
            module-level active backend (set with
            :func:`ceng.backends.set_backend`).
        tokenizer: Optional ``tiktoken`` encoding for accurate
            budgeting.
        summary_max_tokens: Per-leaf summary target. Defaults to
            ``budget_tokens // 4``.
        partition_max_tokens: Per-leaf partition ceiling. Defaults to
            ``summary_max_tokens * 2``.
        **call_kw: Forwarded to ``backend.complete`` for every call
            (``temperature``, ``max_tokens``, etc.).

    Returns:
        A new message list with the user-message replaced by the
        compressed text. Use :class:`CompressResult` (returned from
        :func:`compress_with_stats`) when you also want stats.
    """
    result = compress_with_stats(
        messages,
        budget_tokens=budget_tokens,
        llm=llm,
        cache_dir=cache_dir,
        backend=backend,
        tokenizer=tokenizer,
        summary_max_tokens=summary_max_tokens,
        partition_max_tokens=partition_max_tokens,
        **call_kw,
    )
    return result.messages


def compress_with_stats(
    messages: list[dict],
    *,
    budget_tokens: int,
    llm: str = "gpt-4o-mini",
    cache_dir: str = ".ceng_cache",
    backend: Optional[Backend] = None,
    tokenizer: Any = None,
    summary_max_tokens: Optional[int] = None,
    partition_max_tokens: Optional[int] = None,
    **call_kw: Any,
) -> CompressResult:
    """Same as :func:`ppa_compress` but also returns the stats block."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    if summary_max_tokens is None:
        summary_max_tokens = max(64, budget_tokens // 4)
    if partition_max_tokens is None:
        partition_max_tokens = max(summary_max_tokens, summary_max_tokens * 2)
    backend = backend or get_backend()
    cache = Cache(cache_dir=cache_dir) if cache_dir else _NullCache()

    target_index, target_text = _pick_target_message(messages)
    original_tokens = count_tokens(target_text, tokenizer)
    if original_tokens <= budget_tokens:
        return CompressResult(
            messages=[dict(m) for m in messages],
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            leaf_count=0,
            cache_hits=0,
            cache_misses=0,
            backend=backend,
            model=llm,
        )

    leaves = partition_text(target_text, max_tokens=partition_max_tokens, tokenizer=tokenizer)
    if not leaves:
        return _no_op_result(messages, target_text, original_tokens, backend, llm)
    if len(leaves) == 1:
        compressed = _summarize_leaf(
            backend=backend,
            model=llm,
            leaf=leaves[0].text,
            target_tokens=summary_max_tokens,
            cache=cache,
            tokenizer=tokenizer,
            call_kw=call_kw,
        )
        cache_hits = 1 if compressed.cache_hit else 0
        cache_misses = 0 if compressed.cache_hit else 1
        messages = _replace_message(messages, target_index, _wrap_compressed(compressed.text, target_text))
        return CompressResult(
            messages=messages,
            original_tokens=original_tokens,
            compressed_tokens=count_tokens(compressed.text, tokenizer),
            leaf_count=1,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            backend=backend,
            model=llm,
        )

    summarised = [
        _summarize_leaf(
            backend=backend,
            model=llm,
            leaf=leaf.text,
            target_tokens=summary_max_tokens,
            cache=cache,
            tokenizer=tokenizer,
            call_kw=call_kw,
        )
        for leaf in leaves
    ]
    hits = sum(1 for s in summarised if s.cache_hit)
    misses = len(summarised) - hits
    combined_text = _combine_summaries(
        backend=backend,
        model=llm,
        summaries=[s.text for s in summarised],
        target_tokens=summary_max_tokens,
        cache=cache,
        call_kw=call_kw,
    )
    messages = _replace_message(
        messages, target_index, _wrap_compressed(combined_text, target_text)
    )
    return CompressResult(
        messages=messages,
        original_tokens=original_tokens,
        compressed_tokens=count_tokens(combined_text, tokenizer),
        leaf_count=len(leaves),
        cache_hits=hits,
        cache_misses=misses,
        backend=backend,
        model=llm,
    )


@dataclass
class _LeafResult:
    """Result of summarising one leaf; tracks whether the cache served it."""

    text: str
    cache_hit: bool


def _summarize_leaf(
    *,
    backend: Backend,
    model: str,
    leaf: str,
    target_tokens: int,
    cache: Any,
    tokenizer: Any,
    call_kw: dict,
) -> _LeafResult:
    """Summarise one leaf, consulting the cache first."""
    cache_key = make_key(
        {
            "op": "summarize",
            "model": model,
            "version": PROMPT_VERSION,
            "leaf_sha256": hashlib.sha256(leaf.encode("utf-8")).hexdigest(),
            "target_tokens": target_tokens,
        }
    )
    cached = cache.get(NAMESPACE_SUMMARIZE, cache_key) if cache else None
    if cached is not None:
        return _LeafResult(text=str(cached["text"]), cache_hit=True)
    text = _call_summarize(
        backend=backend, model=model, leaf=leaf, target_tokens=target_tokens, call_kw=call_kw
    )
    if cache:
        cache.set(NAMESPACE_SUMMARIZE, cache_key, {"text": text})
    return _LeafResult(text=text, cache_hit=False)


def _combine_summaries(
    *,
    backend: Backend,
    model: str,
    summaries: list[str],
    target_tokens: int,
    cache: Any,
    call_kw: dict,
) -> str:
    """Combine a list of leaf summaries into one coherent summary."""
    cache_key = (
        make_key(
            {
                "op": "combine",
                "model": model,
                "version": PROMPT_VERSION,
                "summaries_sha256": hashlib.sha256(
                    "\n\n".join(summaries).encode("utf-8")
                ).hexdigest(),
                "target_tokens": target_tokens,
            }
        )
        if cache
        else None
    )
    if cache and cache_key is not None:
        cached = cache.get(NAMESPACE_SUMMARIZE, cache_key)
        if cached is not None:
            return str(cached["text"])
    text = _call_combine(
        backend=backend, model=model, summaries=summaries, target_tokens=target_tokens, call_kw=call_kw
    )
    if cache and cache_key is not None:
        cache.set(NAMESPACE_SUMMARIZE, cache_key, {"text": text})
    return text


def _call_summarize(
    *, backend: Backend, model: str, leaf: str, target_tokens: int, call_kw: dict
) -> str:
    """Single leaf-summarise LLM call."""
    prompt = (
        f"Summarise the following text in roughly {target_tokens} tokens. "
        "Preserve every unique fact, entity, number, and proper noun. "
        "Return only the summary.\n\n"
        f"TEXT:\n{leaf}"
    )
    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", max(64, target_tokens * 2))
    return backend.complete(
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=model,
        **kw,
    )


def _call_combine(
    *, backend: Backend, model: str, summaries: list[str], target_tokens: int, call_kw: dict
) -> str:
    """Single combine-over-summaries LLM call."""
    body = "\n\n---\n\n".join(summaries)
    prompt = (
        f"Combine the following {len(summaries)} section summaries into ONE "
        f"coherent summary of roughly {target_tokens} tokens. Preserve every "
        "unique fact, entity, number, and proper noun from every section. "
        "Return only the combined summary.\n\n"
        f"SECTIONS:\n{body}"
    )
    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", max(128, target_tokens * 2))
    return backend.complete(
        messages=[
            {"role": "system", "content": COMBINE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=model,
        **kw,
    )


def _pick_target_message(messages: list[dict]) -> tuple[int, str]:
    """Pick the longest user-role message; fall back to the last message."""
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    best_index = -1
    best_text = ""
    for idx in user_indices:
        content = messages[idx].get("content")
        text = content if isinstance(content, str) else _flatten_content(content)
        if text and len(text) > len(best_text):
            best_index = idx
            best_text = text
    if best_index == -1 and messages:
        last = messages[-1]
        best_index = len(messages) - 1
        best_text = last.get("content", "") if isinstance(last.get("content"), str) else ""
    if not best_text:
        raise ValueError("no user-role text found in messages to compress")
    return best_index, best_text


def _flatten_content(content: Any) -> str:
    """Flatten an OpenAI-style list-of-parts content into one string."""
    if not isinstance(content, list):
        return str(content)
    return "".join(
        c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")
    )


def _replace_message(messages: list[dict], index: int, new_content: str) -> list[dict]:
    """Replace ``messages[index].content`` with ``new_content``, deep-copying."""
    out = [dict(m) for m in messages]
    out[index] = dict(out[index])
    out[index]["content"] = new_content
    return out


def _wrap_compressed(summary: str, original: str) -> str:
    """Wrap a compressed summary in a header so provenance is preserved."""
    header = "[PPA-compressed summary of the following context. Original below for reference.]\n\n"
    footer = "\n\n[Original]\n" + original
    return header + summary + footer


def _no_op_result(
    messages: list[dict],
    target_text: str,
    original_tokens: int,
    backend: Backend,
    model: str,
) -> CompressResult:
    """Return a no-op result when partitioning yields zero leaves."""
    return CompressResult(
        messages=[dict(m) for m in messages],
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        leaf_count=0,
        cache_hits=0,
        cache_misses=0,
        backend=backend,
        model=model,
    )


class _NullCache:
    """No-op cache used when ``cache_dir`` is empty."""

    def get(self, *_args: Any, **_kw: Any) -> None:
        return None

    def set(self, *_args: Any, **_kw: Any) -> None:
        return None
