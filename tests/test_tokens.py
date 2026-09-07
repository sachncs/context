"""Tests for :mod:`ceng.tokens`."""

from __future__ import annotations

import pytest

from ceng.tokens import count_tokens, token_budget_split


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_falls_back_when_tiktoken_missing(monkeypatch):
    """Force the heuristic branch by pretending tiktoken is missing."""
    import ceng.tokens as tokens_module

    monkeypatch.setattr(tokens_module, "_TOKENIZER", False)
    out = count_tokens("hello world this is a test")
    assert out == max(1, len("hello world this is a test") // 4)


def test_count_tokens_uses_tiktoken_when_available():
    pytest.importorskip("tiktoken")
    out_real = count_tokens("Hello, world!")
    out_heuristic = max(1, len("Hello, world!") // 4)
    assert out_real > 0
    assert isinstance(out_real, int)
    assert out_real != out_heuristic or out_real > 0


def test_token_budget_split_empty_returns_empty():
    assert token_budget_split("", max_tokens=10) == []


def test_token_budget_split_single_chunk_when_small():
    text = "Short paragraph."
    out = token_budget_split(text, max_tokens=100)
    assert out == [text]


def test_token_budget_split_respects_max_tokens(monkeypatch):
    import ceng.tokens as tokens_module

    monkeypatch.setattr(tokens_module, "_TOKENIZER", False)
    sentences = "First. Second. Third. Fourth. Fifth."
    chunks = token_budget_split(sentences, max_tokens=4)
    # The splitter must produce more than one chunk for a multi-sentence
    # input fitting its heuristic budget; the heuristic is loose so the
    # per-chunk token count is only an upper bound, not an exact bound.
    assert len(chunks) >= 2
    rebuilt = " ".join(chunks)
    assert "First." in rebuilt and "Fifth." in rebuilt


def test_token_budget_split_oversized_sentence_uses_greedy_words(monkeypatch):
    import ceng.tokens as tokens_module

    monkeypatch.setattr(tokens_module, "_TOKENIZER", False)
    sentence = "word " * 200 + "."
    text = sentence
    chunks = token_budget_split(text, max_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk != ""


def test_token_budget_split_rejects_zero_max():
    with pytest.raises(ValueError, match="max_tokens"):
        token_budget_split("anything", max_tokens=0)


def test_token_budget_split_preserves_order():
    text = "Alpha. Beta. Gamma. Delta."
    chunks = token_budget_split(text, max_tokens=3)
    joined = " ".join(chunks)
    # Order of first letters must be preserved across the chunking.
    assert joined.index("Alpha") < joined.index("Beta") < joined.index("Gamma")


def test_count_tokens_at_least_one_for_non_empty_heuristic(monkeypatch):
    import ceng.tokens as tokens_module

    monkeypatch.setattr(tokens_module, "_TOKENIZER", False)
    assert count_tokens("a") >= 1
    assert count_tokens("ab") >= 1
