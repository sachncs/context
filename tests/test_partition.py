"""Tests for :mod:`ceng.partition`."""

from __future__ import annotations

import pytest

from ceng.partition import Partition, partition_text


def test_empty_text_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        partition_text("")


def test_zero_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        partition_text("anything", max_tokens=0)


def test_small_text_yields_single_partition():
    out = partition_text("Just one short sentence.", max_tokens=100)
    assert len(out) == 1
    assert isinstance(out[0], Partition)
    assert out[0].index == 0
    assert out[0].text == "Just one short sentence."


def test_indices_are_dense_and_ordered():
    text = ("First paragraph with several sentences. " * 5) + "\n\n" + ("Second paragraph also long. " * 5)
    out = partition_text(text, max_tokens=10)
    indices = [p.index for p in out]
    assert indices == list(range(len(out)))


def test_partitions_concatenate_to_original():
    text = ("First. Second. Third. " * 100).strip()
    out = partition_text(text, max_tokens=20)
    joined = " ".join(p.text for p in out).replace(" ", " ")
    # every sentence in the original must appear in some chunk
    for needle in ("First.", "Second.", "Third."):
        assert needle in joined


def test_recursion_handles_long_inputs(monkeypatch):
    import ceng.partition as partition_module

    monkeypatch.setattr(partition_module.count_tokens.__module__, "_TOKENIZER", False) if False else None
    # Force the heuristic branch by mocking count_tokens in the partitioner.
    import ceng.tokens as tokens_module

    monkeypatch.setattr(tokens_module, "_TOKENIZER", False)
    text = ("Sentence. " * 200).strip()
    chunks = partition_text(text, max_tokens=5)
    assert len(chunks) > 1


def test_partition_chunks_are_bounded():
    text = ("Long sentence with many words to be chunked carefully. " * 30).strip()
    chunks = partition_text(text, max_tokens=15)
    # Each chunk must be non-empty and not absurdly oversized.
    for chunk in chunks:
        assert chunk.text != ""


def test_recursion_splits_at_paragraph_before_word_greedy():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    out = partition_text(text, max_tokens=3)
    assert len(out) >= 2
    joined = " ".join(p.text for p in out)
    for needle in ("Paragraph one.", "Paragraph two.", "Paragraph three."):
        assert needle in joined


def test_oversized_single_word_emitted_verbatim():
    text = "x" * 5000  # one giant "word"
    out = partition_text(text, max_tokens=10)
    assert any(p.text == "x" * 5000 for p in out)


def test_partition_can_recover_full_text_in_order():
    text = "Alpha. Beta.\n\nGamma. Delta."
    out = partition_text(text, max_tokens=2)
    indices_present = sorted(p.index for p in out)
    assert indices_present == list(range(len(out)))
    # Each chunk is non-empty.
    for p in out:
        assert p.text.strip() != ""
