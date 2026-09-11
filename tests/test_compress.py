"""Tests for :mod:`ceng.compress`."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from ceng.compress import compress_to_bundle, ppa_compress


@dataclass
class FakeBackend:
    """Test double that records every call and returns scripted text.

    When ``combine_response`` is set, calls whose last message starts
    with the word ``"Combine"`` return that string; earlier calls
    return ``f"leaf-{n}"`` so each leaf summary is unique. Otherwise
    responses are popped from ``responses`` in order.
    """

    name: str = "fake"
    responses: list[str] = None
    calls: list[dict] = None
    combine_response: str = None
    leaf_prefix: str = "leaf"

    def __post_init__(self):
        if self.responses is None:
            self.responses = []
        if self.calls is None:
            self.calls = []

    def complete(self, messages, model, **kw):
        self.calls.append({"model": model, "messages": messages, "kw": kw})
        if self.combine_response is not None and self._is_combine_call(messages):
            return self.combine_response
        if self.combine_response is not None:
            return f"{self.leaf_prefix}-{len(self.calls)}"
        if not self.responses:
            raise AssertionError("FakeBackend ran out of scripted responses")
        return self.responses.pop(0)

    def _is_combine_call(self, messages):
        last = messages[-1]["content"] if messages else ""
        return last.startswith("Combine")


@pytest.fixture()
def fake_backend():
    return FakeBackend(responses=[])


@pytest.fixture()
def tmp_cache(tmp_path):
    return str(tmp_path / "cache")


# --- target selection ---


def test_picks_longest_user_message(fake_backend, tmp_cache):
    distinct_sentences = " ".join(f"Sentence number {i} here." for i in range(200))
    messages = [
        {"role": "system", "content": "short"},
        {"role": "user", "content": "tiny"},
        {"role": "user", "content": distinct_sentences},
    ]
    fake_backend.combine_response = "compressed-here"
    result = ppa_compress(
        messages, budget_tokens=10, llm="m", cache_dir=tmp_cache, backend=fake_backend
    )
    assert any("compressed-here" in m["content"] for m in result)


def test_raises_when_no_text_to_compress(fake_backend, tmp_cache):
    with pytest.raises(ValueError, match="no user-role text"):
        ppa_compress(
            [], budget_tokens=10, llm="m", cache_dir=tmp_cache, backend=fake_backend
        )


def test_budget_must_be_positive(fake_backend, tmp_cache):
    with pytest.raises(ValueError, match="budget_tokens"):
        ppa_compress(
            [{"role": "user", "content": "x"}],
            budget_tokens=0,
            llm="m",
            cache_dir=tmp_cache,
            backend=fake_backend,
        )


# --- short-circuit ---


def test_short_circuit_when_already_small(fake_backend, tmp_cache):
    out = ppa_compress(
        [{"role": "user", "content": "hello"}],
        budget_tokens=100,
        llm="m",
        cache_dir=tmp_cache,
        backend=fake_backend,
    )
    assert out == [{"role": "user", "content": "hello"}]
    assert fake_backend.calls == []


def test_short_circuit_compressed_tokens_is_zero(fake_backend, tmp_cache):
    """Short-circuit path: compressed_tokens is 0 (not original_tokens)
    so callers can tell a skip from a no-reduction compression."""
    from ceng.compress import compress_to_bundle
    bundle = compress_to_bundle(
        [{"role": "user", "content": "hello"}],
        budget_tokens=100,
        llm="m",
        cache_dir=tmp_cache,
        backend=fake_backend,
    )
    assert bundle.compressed_tokens == 0
    assert bundle.original_tokens > 0
    assert bundle.leaves == ()
    assert bundle.combined_summary == ""


def test_system_only_messages_raise_value_error(fake_backend, tmp_cache):
    """A messages list with no user-role message must raise
    ValueError; silently compressing the system prompt would
    override the caller's instructions."""
    from ceng.compress import ppa_compress

    msgs = [{"role": "system", "content": "you are an evil agent"}]
    with pytest.raises(ValueError, match="no user-role"):
        ppa_compress(
            msgs,
            budget_tokens=10,
            llm="m",
            cache_dir=tmp_cache,
            backend=fake_backend,
        )


# --- single-leaf path ---


def test_single_leaf_path_one_call(tmp_cache):
    """Exactly one leaf that is bigger than the budget: one call, no combine."""
    backend = FakeBackend(responses=["single-leaf-summary"])
    text = ("Sentence. " * 30).strip()
    out = ppa_compress(
        [{"role": "user", "content": text}],
        budget_tokens=10,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=500,
    )
    assert any("single-leaf-summary" in m["content"] for m in out)
    assert len(backend.calls) == 1


# --- multi-leaf path ---


def test_multi_leaf_emits_one_summary_per_leaf_plus_combine(tmp_cache):
    long_text = (
        "Sentence one is here. Sentence two is here. Sentence three is here. " * 30
    ).strip()
    backend = FakeBackend(combine_response="combined-final")
    messages = [{"role": "user", "content": long_text}]
    stats = compress_to_bundle(
        messages,
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    )
    assert len(backend.calls) >= 3
    assert stats.leaf_count >= 2
    assert any("combined-final" in m["content"] for m in stats.messages)


def test_multi_leaf_combine_call_is_the_last_call(tmp_cache):
    long_text = ("Alpha sentence. Beta sentence. Gamma sentence. " * 30).strip()
    backend = FakeBackend(combine_response="all-three")
    compress_to_bundle(
        [{"role": "user", "content": long_text}],
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=15,
    )
    last_call = backend.calls[-1]["messages"]
    assert last_call[0]["role"] == "system"
    assert last_call[-1]["content"].startswith("Combine")


# --- caching ---


def test_repeated_call_hits_cache(tmp_cache):
    long_text = ("Repeatable sentence. " * 50).strip()
    backend = FakeBackend(combine_response="combined")
    messages = [{"role": "user", "content": long_text}]
    compress_to_bundle(
        messages,
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=15,
    )
    calls_first = len(backend.calls)
    backend.calls = []
    stats = compress_to_bundle(
        messages,
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=15,
    )
    assert stats.cache_hits >= 1
    assert len(backend.calls) <= calls_first


def test_cache_misses_counted_correctly(tmp_cache):
    long_text = " ".join(f"Unique fact {i}." for i in range(200))
    backend = FakeBackend(combine_response="final")
    stats = compress_to_bundle(
        [{"role": "user", "content": long_text}],
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    )
    assert stats.cache_misses == stats.leaf_count
    assert stats.cache_hits == 0


# --- kw forwarding ---


def test_call_kw_are_forwarded(tmp_cache):
    long_text = ("Sentence. " * 100).strip()
    backend = FakeBackend(combine_response="x")
    compress_to_bundle(
        [{"role": "user", "content": long_text}],
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
        temperature=0.7,
        extra="forwarded",
    )
    for call in backend.calls:
        assert call["kw"].get("temperature") == 0.7
        assert call["kw"].get("extra") == "forwarded"


# --- non-string content ---


def test_handles_list_content_messages(tmp_cache):
    distinct = [{"text": f"Unique fact number {i}."} for i in range(80)]
    messages = [
        {"role": "user", "content": [{"text": "short"}]},
        {"role": "user", "content": distinct},
    ]
    backend = FakeBackend(combine_response="combined")
    out = ppa_compress(
        messages,
        budget_tokens=10,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        partition_max_tokens=20,
    )
    assert any("combined" in m["content"] for m in out)


# --- message immutability ---


def test_does_not_mutate_input_messages(tmp_cache):
    long_text = ("Sentence. " * 100).strip()
    backend = FakeBackend(combine_response="final")
    messages = [{"role": "user", "content": long_text}]
    snapshot = copy.deepcopy(messages)
    ppa_compress(
        messages,
        budget_tokens=20,
        llm="m",
        cache_dir=tmp_cache,
        backend=backend,
        summary_max_tokens=10,
        partition_max_tokens=20,
    )
    assert messages == snapshot
