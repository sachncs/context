import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ceng import errors
from ceng.internals import concurrency
from ceng.partition import (
    FixedWindowPartitioner,
    Partitioner,
    RecursivePartitioner,
)
from ceng.tokenizers import HeuristicTokenizer

TOK = HeuristicTokenizer()


@settings(max_examples=150, deadline=None)
@given(
    text=st.text(min_size=1, max_size=3000),
    limit=st.integers(min_value=1, max_value=200),
)
def test_lossless_and_bounded(text, limit):
    parts = RecursivePartitioner(limit).split(text, TOK)
    assert "".join(p.text for p in parts) == text
    assert all(p.tokens <= limit for p in parts)
    assert [p.index for p in parts] == list(range(len(parts)))
    assert all(text[p.start : p.end] == p.text for p in parts)


@settings(max_examples=50, deadline=None)
@given(text=st.text(min_size=1, max_size=2000), limit=st.integers(1, 100))
def test_fixed_lossless(text, limit):
    parts = FixedWindowPartitioner(limit).split(text, TOK)
    assert "".join(p.text for p in parts) == text
    assert all(p.tokens <= limit for p in parts)


def test_preserves_newlines_and_prefers_paragraphs():
    text = "line one\nline two\n\n" + "para two. " * 30
    parts = RecursivePartitioner(20).split(text, TOK)
    assert "".join(p.text for p in parts) == text
    assert parts[0].text.startswith("line one\nline two\n\n")


def test_short_text_single_partition():
    assert len(RecursivePartitioner(512).split("hi", TOK)) == 1


def test_unbroken_text_hard_cut():
    parts = RecursivePartitioner(5).split("x" * 200, TOK)
    assert len(parts) > 1 and all(p.tokens <= 5 for p in parts)


def test_validation():
    for cls in (RecursivePartitioner, FixedWindowPartitioner):
        with pytest.raises(errors.ConfigError):
            cls(0)
        with pytest.raises(errors.ValidationError):
            cls(5).split("", TOK)
    assert "recursive" in Partitioner.registry


def test_gather_bounded_order_and_limit():
    live = {"now": 0, "max": 0}

    def make(i):
        async def work():
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.005)
            live["now"] -= 1
            return i

        return work

    out = asyncio.run(
        concurrency.gather_bounded([make(i) for i in range(10)], 3)
    )
    assert out == list(range(10)) and live["max"] == 3


def test_gather_bounded_fail_fast_cancels():
    finished = []

    async def slow():
        await asyncio.sleep(0.2)
        finished.append(1)

    async def boom():
        raise RuntimeError("x")

    async def go():
        await concurrency.gather_bounded([slow, boom], 5)

    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert finished == []
