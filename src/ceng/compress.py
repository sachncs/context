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

Three entry points:

* :func:`ppa_compress` — return only the compressed message list.
* :func:`compress_with_stats` — return a :class:`CompressResult` with
  cache-hit / cache-miss bookkeeping.
* :func:`compress_to_bundle` — return a :class:`CompressionBundle`
  with per-leaf provenance, suitable for building OKF concepts.

The two LLM calls per chunk are:

* **Leaf summary** — one call per leaf, cache key derived from
  ``(model, text_sha256, prompt_template_version)``.
* **Combine** — one call on the list of leaf summaries.

If a message already fits the budget, every entry point short-circuits
and returns the input untouched (no LLM calls).

References:

* https://arxiv.org/abs/2607.15277 — Partition, Prompt, Aggregate.
* https://arxiv.org/abs/2510.26493 — Context Engineering 2.0.
* https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from ceng.backends import Backend, get_backend
from ceng.cache import NAMESPACE_SUMMARIZE, Cache, make_key
from ceng.okf import (
    CENG_BUNDLE_INDEX,
    CENG_COMBINED_SUMMARY,
    CENG_LEAF_SUMMARY,
    Concept,
    Frontmatter,
    RESERVED_INDEX,
    now_iso,
    write_bundle,
)
from ceng.partition import Partition, partition_text
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


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class LeafArtifact:
    """One leaf's contribution to a :class:`CompressionBundle`.

    Attributes:
        index: Zero-based ordinal within the original document, in
            document order.
        text: The original leaf text before summarisation.
        summary: The leaf-level summary produced by the LLM (or read
            from cache).
        cache_hit: Whether the leaf summary came from cache.
    """

    index: int
    text: str
    summary: str
    cache_hit: bool


@dataclass
class CompressionBundle:
    """Rich compression output exposing per-leaf provenance.

    Use :func:`compress_to_bundle` to construct one, or
    :func:`ppa_compress_to_okf` to also persist it as an OKF bundle.

    Attributes:
        messages: The compressed message list (the same shape as
            :func:`ppa_compress` returns).
        original_text: The pre-compression text of the source message.
        original_tokens: Token count of the source message.
        compressed_tokens: Token count of ``combined_summary``.
        leaves: Per-leaf records in document order. Empty for inputs
            that already fit the budget.
        combined_summary: The aggregated coherent summary produced by
            the combine step. Empty string for no-op short-circuits.
        cache_hits: Total number of leaf summaries served from cache.
        cache_misses: Total number of leaf summaries freshly computed.
        backend: Backend used for every call.
        model: Model id used.
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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


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

    See :mod:`ceng.compress` for the full algorithm. Returns only the
    compressed message list; use :func:`compress_with_stats` when you
    also want cache-hit bookkeeping, or :func:`compress_to_bundle`
    when you want per-leaf provenance.
    """
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
    return bundle.messages


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
    return CompressResult(
        messages=bundle.messages,
        original_tokens=bundle.original_tokens,
        compressed_tokens=bundle.compressed_tokens,
        leaf_count=bundle.leaf_count,
        cache_hits=bundle.cache_hits,
        cache_misses=bundle.cache_misses,
        backend=bundle.backend,
        model=bundle.model,
    )


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

    The bundle carries the same compressed message list as
    :func:`ppa_compress` plus per-leaf provenance so that callers
    (e.g. :func:`ppa_compress_to_okf`) can build persistent
    representations of the work.
    """
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
        return CompressionBundle(
            messages=[dict(m) for m in messages],
            original_text=target_text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            leaves=(),
            combined_summary="",
            cache_hits=0,
            cache_misses=0,
            backend=backend,
            model=llm,
        )

    partitions = partition_text(target_text, max_tokens=partition_max_tokens, tokenizer=tokenizer)
    if not partitions:
        return CompressionBundle(
            messages=[dict(m) for m in messages],
            original_text=target_text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            leaves=(),
            combined_summary="",
            cache_hits=0,
            cache_misses=0,
            backend=backend,
            model=llm,
        )

    leaf_results: list[tuple[Partition, _LeafResult]] = []
    for partition in partitions:
        leaf_results.append(
            (
                partition,
                _summarize_leaf(
                    backend=backend,
                    model=llm,
                    leaf=partition.text,
                    target_tokens=summary_max_tokens,
                    cache=cache,
                    tokenizer=tokenizer,
                    call_kw=call_kw,
                ),
            )
        )

    leaves: list[LeafArtifact] = [
        LeafArtifact(
            index=p.index,
            text=p.text,
            summary=r.text,
            cache_hit=r.cache_hit,
        )
        for p, r in leaf_results
    ]
    hits = sum(1 for leaf in leaves if leaf.cache_hit)
    misses = len(leaves) - hits

    if len(partitions) == 1:
        final_text = leaves[0].summary
    else:
        final_text = _combine_summaries(
            backend=backend,
            model=llm,
            summaries=[leaf.summary for leaf in leaves],
            target_tokens=summary_max_tokens,
            cache=cache,
            call_kw=call_kw,
        )

    new_messages = _replace_message(
        messages, target_index, _wrap_compressed(final_text, target_text)
    )
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
    **call_kw: Any,
) -> list[Concept]:
    """Compress ``messages`` and persist the result as an OKF bundle.

    Each leaf becomes a concept of type :data:`ceng.okf.CENG_LEAF_SUMMARY`,
    the combined summary becomes a concept of type
    :data:`ceng.okf.CENG_COMBINED_SUMMARY`, and a top-level
    ``index.md`` of type :data:`ceng.okf.CENG_BUNDLE_INDEX` lists them
    all with cross-links. When the input already fits the budget the
    call short-circuits and writes only the bundle index pointing at
    the original message.

    Args:
        messages: OpenAI-style chat messages list.
        bundle_dir: Directory under which the bundle subdirectory is
            created. The bundle is written to ``{bundle_dir}/{bundle_name}``.
        bundle_name: Subdirectory name for the bundle. Must be a safe
            filename component.
        budget_tokens: Token budget for the compressed output.
        llm: Model id used for summary and combine calls.
        cache_dir: Path to the on-disk cache.
        backend: Optional :class:`Backend`; defaults to the active one.
        tokenizer: Optional ``tiktoken`` encoding for token budgeting.
        summary_max_tokens: Per-leaf summary target.
        partition_max_tokens: Per-leaf partition ceiling.
        **call_kw: Forwarded to ``backend.complete`` for every call.

    Returns:
        The list of :class:`Concept` instances persisted on disk,
        in bundle order (leaves first, then the combined summary,
        then the index pointing at both).

    Raises:
        ValueError: If arguments are invalid or the path scheme
            doesn't permit the requested bundle layout.
    """
    if not bundle_name or bundle_name != bundle_name.strip():
        raise ValueError("bundle_name must be a non-empty directory name")
    if "/" in bundle_name or "\\" in bundle_name or bundle_name in {".", ".."}:
        raise ValueError(f"bundle_name must not contain path separators: {bundle_name!r}")

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

    root = _safe_join(bundle_dir, bundle_name)
    concepts = _bundle_to_concepts(bundle, bundle_name=bundle_name)
    write_bundle(root, concepts)
    return concepts


# ---------------------------------------------------------------------------
# OKF conversion helpers
# ---------------------------------------------------------------------------


def _bundle_to_concepts(bundle: CompressionBundle, *, bundle_name: str) -> list[Concept]:
    """Convert a :class:`CompressionBundle` into OKF concepts.

    Produces:

    * one ``leaf-{index}.md`` concept per leaf;
    * one ``combined.md`` concept carrying the final aggregated
      summary, except when there is only one leaf (the leaf is its
      own summary);
    * one :data:`RESERVED_INDEX` concept listing every output with
      cross-links.

    When the bundle has no leaves (the input already fit the budget),
    only the index is written; the index body explains why.
    """
    concepts: list[Concept] = []

    if bundle.leaves:
        leaf_links: list[str] = []
        for leaf in bundle.leaves:
            path = f"leaf-{leaf.index}.md"
            leaf_links.append(f"[{path}]({path})")
            concepts.append(
                Concept(
                    frontmatter=Frontmatter(
                        type=CENG_LEAF_SUMMARY,
                        title=f"Leaf {leaf.index} summary",
                        description=leaf.summary.splitlines()[0][:200] if leaf.summary else "",
                        tags=(
                            "ppa",
                            f"leaf-{leaf.index}",
                            "cache-hit" if leaf.cache_hit else "cache-miss",
                        ),
                        timestamp=now_iso(),
                    ),
                    body=_render_leaf_body(leaf, bundle_name),
                    path=path,
                )
            )

        if len(bundle.leaves) > 1:
            combined_path = "combined.md"
            concepts.append(
                Concept(
                    frontmatter=Frontmatter(
                        type=CENG_COMBINED_SUMMARY,
                        title=f"{bundle_name} combined summary",
                        description=bundle.combined_summary.splitlines()[0][:200]
                        if bundle.combined_summary
                        else "",
                        tags=("ppa", "combined"),
                        timestamp=now_iso(),
                    ),
                    body=bundle.combined_summary
                    + "\n\n## Cross-links\n\n"
                    + "\n".join(f"- {link}" for link in leaf_links)
                    + "\n",
                    path=combined_path,
                )
            )
            index_entries = ["- [Combined summary](combined.md)"] + [
                f"- {link}" for link in leaf_links
            ]
        else:
            index_entries = [f"- {link}" for link in leaf_links]

        index_body = (
            f"# {bundle_name}\n\n"
            f"Bundle produced by `ceng.ppa_compress_to_okf`. "
            f"{bundle.leaf_count} leaves, "
            f"{bundle.cache_hits} cache hits, {bundle.cache_misses} misses.\n\n"
            f"## Sections\n\n"
            + "\n".join(index_entries)
            + "\n"
        )
        concepts.append(
            Concept(
                frontmatter=Frontmatter(
                    type=CENG_BUNDLE_INDEX,
                    title=f"{bundle_name} index",
                    description="Top-level OKF index generated by ceng",
                    tags=("ppa", "bundle-index"),
                    timestamp=now_iso(),
                ),
                body=index_body,
                path=RESERVED_INDEX,
            )
        )
    else:
        # no compression happened — still emit an index so consumers
        # know the bundle is intentionally small.
        concepts.append(
            Concept(
                frontmatter=Frontmatter(
                    type=CENG_BUNDLE_INDEX,
                    title=f"{bundle_name} index",
                    description="No compression was needed; original already fits the budget.",
                    tags=("ppa", "bundle-index", "noop"),
                    timestamp=now_iso(),
                ),
                body=(
                    f"# {bundle_name}\n\n"
                    f"Original message was {bundle.original_tokens} tokens, "
                    f"under the budget of compressed output. No leaves generated.\n"
                ),
                path=RESERVED_INDEX,
            )
        )
    return concepts


def _render_leaf_body(leaf: LeafArtifact, bundle_name: str) -> str:
    """Build the markdown body for a leaf concept."""
    return (
        f"## Summary\n\n"
        f"{leaf.summary}\n\n"
        f"## Original chunk\n\n"
        f"```\n{leaf.text}\n```\n"
    )


def _safe_join(root: str, child: str) -> str:
    """Join ``root`` and ``child`` and verify the result resolves inside ``root``."""
    from pathlib import Path

    base = Path(root).resolve()
    full = (base / child).resolve()
    if not str(full).startswith(str(base)):
        raise ValueError(f"bundle path {full} escapes root {base}")
    return str(full)


# ---------------------------------------------------------------------------
# Private LLM call helpers
# ---------------------------------------------------------------------------


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


class _NullCache:
    """No-op cache used when ``cache_dir`` is empty."""

    def get(self, *_args: Any, **_kw: Any) -> None:
        return None

    def set(self, *_args: Any, **_kw: Any) -> None:
        return None
