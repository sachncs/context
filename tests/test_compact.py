"""Tests for :mod:`ceng.compact`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ceng.compact import CompactionProvenance, compact_messages


@dataclass
class FakeBackend:
    name: str = "fake"

    def __init__(self, response: str = "summary text"):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, messages, model, **kw):
        self.calls.append({"messages": messages, "model": model})
        return self.response


def test_short_history_returns_unchanged():
    msgs = [{"role": "user", "content": "hello"}]
    out, prov = compact_messages(msgs, backend=FakeBackend())
    assert out == msgs
    assert prov.preserved_first == 1
    assert prov.preserved_last == 0
    assert prov.summarised_count == 0


def test_keep_verbatim_when_preserve_first_plus_preserve_last_exceeds_length():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out, prov = compact_messages(
        msgs, backend=FakeBackend(), preserve_first=2, preserve_last=2
    )
    assert out == msgs
    assert prov.summarised_count == 0


def test_u_shape_preserves_first_and_last_middle_summarised():
    backend = FakeBackend(response="(summary of middle)")
    msgs = [
        {"role": "system", "content": "sys"},  # kept at position 0
        {"role": "user", "content": "head-1"},  # preserved head (preserve_first=1 means system only? no: head slice is messages[:preserve_first])
        {"role": "user", "content": "mid-1"},
        {"role": "user", "content": "mid-2"},
        {"role": "user", "content": "tail-1"},
        {"role": "user", "content": "tail-2"},
    ]
    out, prov = compact_messages(
        msgs,
        backend=backend,
        preserve_first=2,  # system + head-1
        preserve_last=2,  # tail-1 + tail-2
    )
    assert prov.summarised_count == 2  # mid-1, mid-2
    assert prov.preserved_first == 2
    assert prov.preserved_last == 2
    # Final shape: [system, head-1, SUMMARY, tail-1, tail-2]
    assert len(out) == 5
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "head-1"}
    assert out[2]["role"] == "assistant"
    assert "summary" in out[2]["content"].lower()
    assert out[3] == {"role": "user", "content": "tail-1"}
    assert out[4] == {"role": "user", "content": "tail-2"}


def test_summarise_middle_false_drops_middle():
    backend = FakeBackend()
    msgs = [{"role": "user", "content": str(i)} for i in range(20)]
    out, prov = compact_messages(
        msgs, backend=backend, preserve_first=2, preserve_last=3,
        summarise_middle=False,
    )
    assert len(out) == 5
    assert prov.summarised_count == 15
    # No backend call should have happened
    assert backend.calls == []


def test_system_message_in_middle_strips_raises():
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "system", "content": "INSTRUCTIONS"},  # in middle
        {"role": "user", "content": "u2"},
    ]
    with pytest.raises(ValueError, match="system-role"):
        compact_messages(msgs, backend=FakeBackend(), preserve_first=1, preserve_last=1)


def test_system_message_in_preserved_head_is_safe():
    msgs = [
        {"role": "system", "content": "INSTRUCTIONS"},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
    ]
    out, prov = compact_messages(
        msgs, backend=FakeBackend(), preserve_first=2, preserve_last=0,
        summarise_middle=False,
    )
    assert out[0]["content"] == "INSTRUCTIONS"


def test_empty_messages_raises():
    with pytest.raises(ValueError, match="empty"):
        compact_messages([], backend=FakeBackend())


def test_backend_error_carries_strip_length():
    class FailBackend:
        name = "fail"

        def complete(self, messages, model, **kw):
            raise ConnectionError("upstream died")

    msgs = [
        {"role": "user", "content": str(i)} for i in range(10)
    ]
    with pytest.raises(RuntimeError, match="middle-strip summarisation"):
        compact_messages(
            msgs, backend=FailBackend(), preserve_first=1, preserve_last=1
        )


def test_empty_summary_output_rejected():
    class EmptyBackend:
        name = "empty"

        def complete(self, messages, model, **kw):
            return "  "

    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    with pytest.raises(RuntimeError, match="empty"):
        compact_messages(
            msgs, backend=EmptyBackend(), preserve_first=1, preserve_last=1
        )


def test_cache_reuses_summary_when_strip_unchanged(tmp_path):
    """Same backend called twice with the same middle strip; the
    second call hits the cache and never invokes the backend."""
    backend_a = FakeBackend(response="first summary")
    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    out_a, _ = compact_messages(
        msgs,
        backend=backend_a,
        preserve_first=1,
        preserve_last=1,
        cache_dir=str(tmp_path / "cache"),
    )
    backend_b = FakeBackend(response="second summary (should not be used)")
    out_b, _ = compact_messages(
        msgs,
        backend=backend_b,
        preserve_first=1,
        preserve_last=1,
        cache_dir=str(tmp_path / "cache"),
    )
    assert out_a[1]["content"] == out_b[1]["content"]
    assert backend_a.calls
    assert backend_b.calls == []


def test_provenance_carries_counts():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(15)]
    backend = FakeBackend()
    out, prov = compact_messages(
        msgs, backend=backend, preserve_first=2, preserve_last=3,
        summarise_middle=False,
    )
    assert isinstance(prov, CompactionProvenance)
    assert prov.preserved_first == 2
    assert prov.preserved_last == 3
    assert prov.summarised_count == 10


def test_summarised_tokens_uses_count_tokens_not_split():
    from ceng.tokens import count_tokens

    class LongSummaryBackend:
        name = "long"

        def complete(self, messages, model, **kw):
            return "one two three four five six seven eight nine ten"

    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    _, prov = compact_messages(
        msgs,
        backend=LongSummaryBackend(),
        preserve_first=1,
        preserve_last=1,
    )
    expected = count_tokens("one two three four five six seven eight nine ten")
    assert prov.summarised_tokens == expected
    assert prov.summarised_tokens != len(
        "one two three four five six seven eight nine ten".split()
    ) or expected == 10
