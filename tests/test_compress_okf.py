"""Tests for :func:`ceng.compress.ppa_compress_to_okf`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ceng.compress import ppa_compress_to_okf
from ceng.okf import (
    CENG_BUNDLE_INDEX,
    CENG_COMBINED_SUMMARY,
    CENG_LEAF_SUMMARY,
    RESERVED_INDEX,
    find_concept,
    read_bundle,
)


@dataclass
class FakeBackend:
    name: str = "fake"
    responses: list[str] = None
    calls: list[dict] = None
    combine_response: str = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = []
        if self.calls is None:
            self.calls = []

    def complete(self, messages, model, **kw):
        self.calls.append({"model": model, "messages": messages, "kw": kw})
        last = messages[-1]["content"] if messages else ""
        if self.combine_response is not None and last.startswith("Combine"):
            return self.combine_response
        if self.combine_response is not None:
            return f"leaf-{len(self.calls)}"
        if not self.responses:
            raise AssertionError("FakeBackend ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture()
def tmp_cache(tmp_path):
    return str(tmp_path / "cache")


# --- short-circuit (no compression) ---


def test_short_circuit_writes_only_index(tmp_cache, tmp_path):
    backend = FakeBackend(responses=["unused"])
    bundle_root = tmp_path / "ctx-out"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": "tiny"}],
        bundle_dir=str(bundle_root),
        budget_tokens=100,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
    index_only=False, )
    assert len(backend.calls) == 0
    # exactly one concept, the index
    assert len(concepts) == 1
    assert concepts[0].frontmatter.type == CENG_BUNDLE_INDEX
    assert concepts[0].path == Path(RESERVED_INDEX)
    bundle_root = bundle_root / "ppa-context"
    assert (bundle_root / RESERVED_INDEX).exists()


# --- single-leaf path ---


def test_single_leaf_path_writes_leaf_and_index(tmp_cache, tmp_path):
    backend = FakeBackend(responses=["single-leaf-summary"])
    bundle_root = tmp_path / "ctx"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": ("Sentence. " * 30).strip()}],
        bundle_dir=str(bundle_root),
        budget_tokens=10,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=500,
    index_only=False, )
    paths = {c.path.as_posix() for c in concepts}
    # single leaf + index
    assert paths == {"leaf-0.md", RESERVED_INDEX}
    assert concepts[0].frontmatter.type == CENG_LEAF_SUMMARY
    assert concepts[1].frontmatter.type == CENG_BUNDLE_INDEX


# --- multi-leaf path ---


def test_multi_leaf_writes_leaves_combined_and_index(tmp_cache, tmp_path):
    long_text = " ".join(f"Unique fact number {i}." for i in range(120))
    backend = FakeBackend(combine_response="combined-final")
    bundle_root = tmp_path / "ctx"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    types = [c.frontmatter.type for c in concepts]
    leaf_count = sum(t == CENG_LEAF_SUMMARY for t in types)
    assert leaf_count >= 2
    assert types[-1] == CENG_BUNDLE_INDEX
    # The combined concept comes right before the index
    combined = [t for t in types if t == CENG_COMBINED_SUMMARY]
    assert len(combined) == 1
    # The combined concept's path is combined.md
    combined_path = next(c.path.as_posix() for c in concepts if c.path.as_posix() == "combined.md")
    assert combined_path == "combined.md"


def test_okf_index_links_every_leaf(tmp_cache, tmp_path):
    long_text = " ".join(f"Fact number {i}." for i in range(120))
    backend = FakeBackend(combine_response="combined-here")
    bundle_root = tmp_path / "ctx"
    ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    bundle_root = bundle_root / "ppa-context"
    loaded = read_bundle(bundle_root)
    index = find_concept(loaded, RESERVED_INDEX)
    assert index is not None
    # index body references both combined.md and every leaf
    body = index.body
    assert "combined.md" in body
    assert "leaf-0.md" in body


def test_okf_leaves_have_dense_indices(tmp_cache, tmp_path):
    long_text = " ".join(f"Sentence {i}." for i in range(150))
    backend = FakeBackend(combine_response="all-three")
    bundle_root = tmp_path / "ctx"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    leaf_concepts = [
        c for c in concepts if c.frontmatter.type == CENG_LEAF_SUMMARY
    ]
    # dense 0..N-1 in document order (NOT lexical order, which puts
    # leaf-10.md before leaf-2.md)
    indices = [c.path.name for c in leaf_concepts]
    assert indices == [f"leaf-{i}.md" for i in range(len(leaf_concepts))]


# --- bundle round-trip ---


def test_okf_bundle_written_files_round_trip(tmp_cache, tmp_path):
    long_text = " ".join(f"Distinguishable fact {i}." for i in range(100))
    backend = FakeBackend(combine_response="round-trip-combined")
    bundle_root = tmp_path / "ctx"
    written = ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        bundle_name="mybundle",
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    # the function writes to bundle_dir/bundle_name; bundle_name subdir
    # must exist with the written files
    target = bundle_root / "mybundle"
    assert target.is_dir()
    found = {p.name for p in target.rglob("*.md")}
    expected_names = {c.path.name for c in written}
    assert expected_names.issubset(found)

    reloaded = read_bundle(target)
    types = {c.frontmatter.type for c in reloaded}
    assert CENG_LEAF_SUMMARY in types
    assert CENG_COMBINED_SUMMARY in types
    assert CENG_BUNDLE_INDEX in types


# --- argument validation ---


def test_rejects_empty_bundle_name(tmp_cache, tmp_path):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="bundle_name"):
        ppa_compress_to_okf(
            [{"role": "user", "content": "tiny"}],
            bundle_dir=str(tmp_path),
            bundle_name="",
            budget_tokens=100,
            llm="m",
            cache_dir=tmp_cache,
            backend=backend,
        index_only=False, )


def test_rejects_path_separator_in_bundle_name(tmp_cache, tmp_path):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="path separators"):
        ppa_compress_to_okf(
            [{"role": "user", "content": "tiny"}],
            bundle_dir=str(tmp_path),
            bundle_name="a/b",
            budget_tokens=100,
            llm="m",
            cache_dir=tmp_cache,
            backend=backend,
        index_only=False, )


def test_rejects_budget_tokens_zero(tmp_cache, tmp_path):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="budget_tokens"):
        ppa_compress_to_okf(
            [{"role": "user", "content": "x"}],
            bundle_dir=str(tmp_path),
            budget_tokens=0,
            llm="m",
            cache_dir=tmp_cache,
            backend=backend,
        index_only=False, )


# --- leaf content reflects original chunk ---


def test_leaf_body_includes_original_chunk_text(tmp_cache, tmp_path):
    long_text = " ".join(f"Original fact number {i}." for i in range(120))
    backend = FakeBackend(combine_response="combined")
    bundle_root = tmp_path / "ctx"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    leaves = [c for c in concepts if c.frontmatter.type == CENG_LEAF_SUMMARY]
    # at least one leaf body should contain an "Original" heading
    assert any("Original" in c.body for c in leaves)
    # the original chunk text lives in a fenced code block
    assert any("```" in c.body for c in leaves)


# --- cache propagates to OKF tags ---


def test_cache_hit_marked_in_tags(tmp_cache, tmp_path):
    long_text = " ".join(f"Sentence {i}." for i in range(120))
    backend = FakeBackend(combine_response="combined-final")
    bundle_root = tmp_path / "ctx"
    ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    # second run — every leaf should now be a cache hit
    backend.calls = []
    backend.combine_response = "combined-final"
    concepts = ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root / "v2"),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    index_only=False, )
    leaf_tags = [
        c.frontmatter.tags
        for c in concepts
        if c.frontmatter.type == CENG_LEAF_SUMMARY
    ]
    # either no cache-miss tags at all, or all cache-hit
    assert all("cache-hit" in tags for tags in leaf_tags)


# --- kwargs forwarded ---


def test_call_kw_are_forwarded_through_okf(tmp_cache, tmp_path):
    long_text = " ".join(f"Sentence {i}." for i in range(120))
    backend = FakeBackend(combine_response="combined")
    bundle_root = tmp_path / "ctx"
    ppa_compress_to_okf(
        [{"role": "user", "content": long_text}],
        bundle_dir=str(bundle_root),
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
        temperature=0.4,
    index_only=False, )
    for call in backend.calls:
        assert call["kw"].get("temperature") == 0.4
