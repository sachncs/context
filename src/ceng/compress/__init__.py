"""Partition-Prompt-Aggregate compression for long contexts.

:func:`ppa_compress` / :func:`compress_to_bundle` take a chat-message
list, find the longest user-role message, partition it into
token-bounded leaves (via :func:`ceng.partition.partition_text`),
summarise each leaf in isolation (cached), then aggregate the leaf
summaries with one final combiner call to produce a single coherent
condensed message.

This is the macro-fallacy-resistant path: instead of asking one
model "please summarise this whole document" (which loses detail),
we ask the model about each chunk and then aggregate the chunk-
level summaries (which preserves more unique facts).

If a message already fits the budget, the pipeline short-circuits
and returns the input untouched (no LLM calls).

Two LLM call patterns:

* **Leaf summary** — one call per leaf, cache key derived from
  ``(model, text_sha256, prompt_template_version)``.
* **Combine** — one call on the list of leaf summaries.

References:

* https://arxiv.org/abs/2607.15277 — Partition, Prompt, Aggregate.
* https://arxiv.org/abs/2510.26493 — Context Engineering 2.0.
* https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ceng.backends import Backend, get_backend
from ceng.cache import NAMESPACE_SUMMARIZE, Cache, make_key
from ceng.compress.bundle import build_bundle_concepts
from ceng.compress.log import logger
from ceng.compress.prompts import (
    COMBINE_SYSTEM,
    PROMPT_VERSION,
    SUMMARIZE_SYSTEM,
    build_combine_prompt,
    build_summarize_prompt,
)
from ceng.okf import Concept, write_bundle
from ceng.partition import partition_text
from ceng.tokens import count_tokens


MAX_LEAVES = 512

WINDOWS_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


class CompressError(RuntimeError):
    """Raised when the compression pipeline fails.

    Carries ``leaf_index`` (which chunk broke; ``None`` for non-leaf
    failures) and ``cause`` so callers can pinpoint and re-run from
    a known good checkpoint.
    """

    def __init__(
        self,
        message: str,
        *,
        leaf_index: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.leaf_index = leaf_index
        self.cause = cause


@dataclass(frozen=True)
class LeafArtifact:
    """One leaf's contribution to a :class:`CompressionBundle`."""

    index: int
    text: str
    summary: str
    cache_hit: bool


@dataclass
class CompressionBundle:
    """Rich compression output exposing per-leaf provenance.

    Use :func:`compress_to_bundle` to construct one, or
    :func:`ppa_compress_to_okf` to also persist it as an OKF
    bundle.

    Attributes:
        compressed_tokens: Token count of the produced summary.
            ``0`` when the call short-circuited (the input already
            fit the budget and no LLM call was made); equal to
            ``original_tokens`` in that case is meaningless because
            nothing was compressed. Callers should check
            ``leaves == ()`` (and/or ``combined_summary == ""``)
            to detect a skip.
    """

    messages: list[dict]
    original_text: str
    original_tokens: int
    compressed_tokens: int
    leaves: tuple[LeafArtifact, ...]
    combined_summary: str
    cache_hits: int
    cache_misses: int
    backend: Backend
    model: str

    @property
    def leaf_count(self) -> int:
        """Number of leaves in the bundle."""
        return len(self.leaves)


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

    Returns only the compressed message list; use
    :func:`compress_to_bundle` for per-leaf provenance or
    :func:`ppa_compress_to_okf` to persist the result as an OKF
    bundle.
    """
    return compress_to_bundle(
        messages,
        budget_tokens=budget_tokens,
        llm=llm,
        cache_dir=cache_dir,
        backend=backend,
        tokenizer=tokenizer,
        summary_max_tokens=summary_max_tokens,
        partition_max_tokens=partition_max_tokens,
        **call_kw,
    ).messages


def compress_to_bundle(
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
) -> CompressionBundle:
    """Compress ``messages`` and return a :class:`CompressionBundle`.

    Args:
        messages: OpenAI-style chat messages list. The longest
            user-role message is the compression target.
        budget_tokens: Token budget for the compressed output. Soft
            target.
        llm: Model id used for summary and combine calls.
        cache_dir: Path to the on-disk cache; pass ``""`` to disable.
        backend: Optional :class:`Backend`; defaults to the active one.
        tokenizer: Optional ``tiktoken`` encoding for accurate counts.
        summary_max_tokens: Per-leaf summary target.
        partition_max_tokens: Per-leaf partition ceiling.
        **call_kw: Forwarded to ``backend.complete`` for every call.

    Returns:
        A :class:`CompressionBundle`.

    Raises:
        CompressError: If more than :data:`MAX_LEAVES` leaves are
            produced (caller asked for too small a budget).
    """
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    if summary_max_tokens is None:
        summary_max_tokens = max(64, budget_tokens // 4)
    if partition_max_tokens is None:
        partition_max_tokens = max(summary_max_tokens, summary_max_tokens * 2)
    backend = backend or get_backend()
    cache: Optional[Cache] = Cache(cache_dir=cache_dir) if cache_dir else None

    target_index, target_text = pick_target_message(messages)
    original_tokens = count_tokens(target_text, tokenizer)
    if original_tokens <= budget_tokens:
        return CompressionBundle(
            messages=[dict(m) for m in messages],
            original_text=target_text,
            original_tokens=original_tokens,
            compressed_tokens=0,
            leaves=(),
            combined_summary="",
            cache_hits=0,
            cache_misses=0,
            backend=backend,
            model=llm,
        )

    partitions = partition_text(
        target_text, max_tokens=partition_max_tokens, tokenizer=tokenizer
    )
    if not partitions:
        return CompressionBundle(
            messages=[dict(m) for m in messages],
            original_text=target_text,
            original_tokens=original_tokens,
            compressed_tokens=0,
            leaves=(),
            combined_summary="",
            cache_hits=0,
            cache_misses=0,
            backend=backend,
            model=llm,
        )

    if len(partitions) > MAX_LEAVES:
        raise CompressError(
            f"partition produced {len(partitions)} leaves; "
            f"raised partition_max_tokens (current={partition_max_tokens}) "
            f"or summary_max_tokens (current={summary_max_tokens})"
        )

    leaves: list[LeafArtifact] = []
    for i, partition in enumerate(partitions):
        try:
            summary, cache_hit = summarise_leaf(
                backend=backend,
                model=llm,
                leaf=partition.text,
                target_tokens=summary_max_tokens,
                cache=cache,
                tokenizer=tokenizer,
                call_kw=call_kw,
            )
        except Exception as exc:
            raise CompressError(
                f"leaf summary failed: {exc!r}",
                leaf_index=i,
                cause=exc,
            ) from exc
        leaves.append(
            LeafArtifact(
                index=partition.index,
                text=partition.text,
                summary=summary,
                cache_hit=cache_hit,
            )
        )

    hits = sum(1 for leaf in leaves if leaf.cache_hit)
    misses = len(leaves) - hits

    if len(partitions) == 1:
        final_text = leaves[0].summary
    else:
        try:
            final_text = combine_summaries(
                backend=backend,
                model=llm,
                summaries=[leaf.summary for leaf in leaves],
                target_tokens=summary_max_tokens,
                cache=cache,
                call_kw=call_kw,
            )
        except Exception as exc:
            raise CompressError(
                f"combine failed: {exc!r}",
                leaf_index=None,
                cause=exc,
            ) from exc

    new_messages = replace_message(messages, target_index, final_text)
    return CompressionBundle(
        messages=new_messages,
        original_text=target_text,
        original_tokens=original_tokens,
        compressed_tokens=count_tokens(final_text, tokenizer),
        leaves=tuple(leaves),
        combined_summary=final_text,
        cache_hits=hits,
        cache_misses=misses,
        backend=backend,
        model=llm,
    )


def ppa_compress_to_okf(
    messages: list[dict],
    *,
    bundle_dir: str,
    bundle_name: str = "ppa-context",
    budget_tokens: int,
    llm: str = "gpt-4o-mini",
    cache_dir: str = ".ceng_cache",
    backend: Optional[Backend] = None,
    tokenizer: Any = None,
    summary_max_tokens: Optional[int] = None,
    partition_max_tokens: Optional[int] = None,
    index_only: bool = True,
    **call_kw: Any,
) -> list[Concept]:
    """Compress ``messages`` and persist the result as an OKF bundle.

    Each leaf becomes a concept of type
    :data:`ceng.okf.CENG_LEAF_SUMMARY`, the combined summary a
    concept of type :data:`ceng.okf.CENG_COMBINED_SUMMARY`, and a
    top-level ``index.md`` of type
    :data:`ceng.okf.CENG_BUNDLE_INDEX` lists them all with
    cross-links.

    When the input already fits the budget the call short-circuits
    and writes only the bundle index pointing at the original message.

    ``index_only=True`` (the default in v0.4.0+; Anthropic
    just-in-time pattern) writes only the ``index.md`` and the
    combined summary file. Per-leaf files are NOT materialised on
    disk; consumers that want them use ``ceng.okf.read_concept_file``
    after asking for one.

    Args:
        messages: OpenAI-style chat messages list.
        bundle_dir: Directory under which the bundle subdirectory is
            created.
        bundle_name: Subdirectory name; validated for Windows-
            reserved names, control characters, and length.
        budget_tokens: Token budget for the compressed output.
        llm: Model id for summary and combine calls.
        cache_dir: Path to the on-disk cache.
        backend: Optional :class:`Backend`.
        tokenizer: Optional ``tiktoken`` encoding.
        summary_max_tokens: Per-leaf summary target.
        partition_max_tokens: Per-leaf partition ceiling.
        index_only: If True, only write ``index.md`` + ``combined.md``.
            Set to False for the v0.3.0 behaviour where every leaf
            is materialised.
        **call_kw: Forwarded to ``backend.complete`` for every call.

    Returns:
        The list of :class:`ceng.okf.Concept` instances written to
        disk, in bundle order.

    Raises:
        ValueError: If arguments are invalid or the path scheme
            doesn't permit the requested bundle layout.
    """
    _validate_bundle_name(bundle_name)
    bundle = compress_to_bundle(
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
    root = Path(bundle_dir) / bundle_name
    concepts = build_bundle_concepts(
        bundle, bundle_name, tokenizer, index_only=index_only
    )
    write_bundle(root, concepts)
    logger.info(
        "wrote OKF bundle with %d concepts to %s (index_only=%s)",
        len(concepts), root, index_only,
    )
    return concepts


# ---------------------------------------------------------------------------
# Helpers (private — module-local, NOT exported)
# ---------------------------------------------------------------------------


def pick_target_message(messages: list[dict]) -> tuple[int, str]:
    """Pick the longest user-role message; fall back to the last message.

    Returns ``(index, text)``. The fallback path also flattens
    list-content (OpenAI-style list-of-parts) so a non-user final
    message isn't silently dropped.
    """
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    best_index = -1
    best_text = ""
    for idx in user_indices:
        text = _flatten_content(messages[idx].get("content"))
        if text and len(text) > len(best_text):
            best_index = idx
            best_text = text
    if best_index == -1 and messages:
        last = messages[-1]
        best_index = len(messages) - 1
        best_text = _flatten_content(last.get("content"))
    if not best_text:
        raise ValueError("no user-role text found in messages to compress")
    return best_index, best_text


def _flatten_content(content: Any) -> str:
    """Flatten an OpenAI-style list-of-parts content into one string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return ""


def replace_message(messages: list[dict], index: int, new_content: str) -> list[dict]:
    """Return a copy of ``messages`` with ``messages[index].content`` replaced.

    The original input is not mutated (deep copy at every level).
    """
    out = [dict(m) for m in messages]
    out[index] = dict(out[index])
    out[index]["content"] = new_content
    return out


def _validate_bundle_name(bundle_name: str) -> None:
    """Reject unsafe bundle-name strings."""
    if not bundle_name or bundle_name != bundle_name.strip():
        raise ValueError("bundle_name must be a non-empty directory name")
    if "/" in bundle_name or "\\" in bundle_name or bundle_name in {".", ".."}:
        raise ValueError(
            f"bundle_name must not contain path separators: {bundle_name!r}"
        )
    if any(ord(c) < 0x20 for c in bundle_name):
        raise ValueError(
            f"bundle_name must not contain control characters: {bundle_name!r}"
        )
    if bundle_name.upper().split(".")[0] in WINDOWS_RESERVED:
        raise ValueError(
            f"bundle_name {bundle_name!r} is reserved on Windows"
        )
    if len(bundle_name) > 100:
        raise ValueError(
            f"bundle_name must be at most 100 chars, got {len(bundle_name)}"
        )


def summarise_leaf(
    *,
    backend: Backend,
    model: str,
    leaf: str,
    target_tokens: int,
    cache: Optional[Cache],
    tokenizer: Any,
    call_kw: dict,
) -> tuple[str, bool]:
    """Summarise one leaf, consulting the cache first.

    Returns ``(summary, cache_hit)``. An empty or whitespace-only
    LLM response raises :class:`CompressError` rather than
    caching poison.
    """
    cache_key = make_key(
        {
            "op": "summarize",
            "model": model,
            "version": PROMPT_VERSION,
            "leaf_sha256": hashlib.sha256(leaf.encode("utf-8")).hexdigest(),
            "target_tokens": target_tokens,
        }
    )
    if cache is not None:
        cached = cache.get(NAMESPACE_SUMMARIZE, cache_key)
        if cached is not None:
            return str(cached["text"]), True
    text = _run_prompt(
        backend=backend,
        model=model,
        system=SUMMARIZE_SYSTEM,
        user=build_summarize_prompt(leaf, target_tokens),
        target_tokens=target_tokens,
        call_kw=call_kw,
    )
    _validate_llm_output(text, context=f"leaf summary (target={target_tokens} tokens)")
    if cache is not None:
        cache.set(NAMESPACE_SUMMARIZE, cache_key, {"text": text})
    return text, False


def combine_summaries(
    *,
    backend: Backend,
    model: str,
    summaries: list[str],
    target_tokens: int,
    cache: Optional[Cache],
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
        if cache is not None
        else None
    )
    if cache is not None and cache_key is not None:
        cached = cache.get(NAMESPACE_SUMMARIZE, cache_key)
        if cached is not None:
            return str(cached["text"])
    text = _run_prompt(
        backend=backend,
        model=model,
        system=COMBINE_SYSTEM,
        user=build_combine_prompt(summaries, target_tokens),
        target_tokens=target_tokens,
        call_kw=call_kw,
    )
    _validate_llm_output(text, context="combine")
    if cache is not None and cache_key is not None:
        cache.set(NAMESPACE_SUMMARIZE, cache_key, {"text": text})
    return text


def _run_prompt(
    *,
    backend: Backend,
    model: str,
    system: str,
    user: str,
    target_tokens: int,
    call_kw: dict,
) -> str:
    """Single LLM call with ceng-standard defaults."""
    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", max(64, target_tokens * 2))
    return backend.complete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        **kw,
    )


def _validate_llm_output(text: str, *, context: str) -> None:
    """Reject empty or whitespace-only LLM output as recoverable cache poison."""
    if not text or not text.strip():
        raise CompressError(
            f"LLM returned empty/whitespace-only output ({context})"
        )
