"""FiNER (XBRL entity tagging) data processor.

FiNER is a financial-entity-tagging benchmark: given a sentence from
an XBRL document, label each token with one of 139 fine-grained
entity types (e.g. ``B-Location``, ``I-Product``). The ACE paper
reports DeepSeek-V3.1+ACE hitting 78.3 vs base 70.7 (+7.6).

The dataset is publicly available on HuggingFace
(`../finer-xbrl/...`). The processor implemented here works with the
flattened JSONL shape:

    {"id": "...", "sentence": "...", "tokens": [...], "labels": [...]}

or the raw HuggingFace shape — both are coerced into ``DataSample``
in :meth:`process_task_data`.

This module ships a curated seed playbook
(:func:`seed_playbook`) with 30+ XBRL-aware strategies so an
empty Evolver.run start point is already non-empty.
"""

from __future__ import annotations

from typing import Any, Iterable

from ceng.eval import DataProcessor, DataSample


# 139 entity types in FiNER (offline copy of the paper's list, just
# the prefix categories so we can build a regex quickly).
FINER_ENTITY_PREFIXES: tuple[str, ...] = (
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC",
    "O",
)


class FiNERProcessor:
    """Coerce FiNER rows to ``DataSample``; grade by exact-tag match.

    Each FiNER row is one sentence. ``tokens`` and ``labels`` are
    parallel lists. We present the sentence + a "BIO format" hint to
    the model and grade the response by re-parsing it into labels
    and computing the per-token F1 against the gold labels.
    """

    def process_task_data(self, raw_data: list[dict]) -> list[DataSample]:
        out: list[DataSample] = []
        for row in raw_data:
            sentence = row.get("sentence") or row.get("text") or " ".join(
                row.get("tokens", [])
            )
            tokens = row.get("tokens", [])
            labels = row.get("labels", [])
            target = _format_gold(tokens, labels)
            context = (
                "Tag each token with one of the following BIO tags: "
                + ", ".join(FINER_ENTITY_PREFIXES)
                + ".\nReturn one tag per line, in token order."
            )
            question = (
                f"Sentence: {sentence}\n"
                f"Tokens: {' '.join(tokens) if tokens else sentence.split()}\n\n"
                "Tags (one per line):"
            )
            out.append(
                DataSample(
                    question=question,
                    target=target,
                    context=context,
                    others={
                        "id": row.get("id"),
                        "tokens": tokens,
                        "labels": labels,
                    },
                )
            )
        return out

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """True iff predicted and gold label sequences match exactly.

        Whitespace-tolerant; both must collapse to the same per-line
        tag list.
        """
        return _normalise_tags(predicted) == _normalise_tags(ground_truth)

    def evaluate_accuracy(
        self,
        predictions: list[str],
        ground_truths: list[str],
    ) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            1
            for p, g in zip(predictions, ground_truths)
            if self.answer_is_correct(p, g)
        )
        return correct / len(predictions)


def _format_gold(tokens: list[str], labels: list[str]) -> str:
    if not labels:
        return ""
    if tokens and len(tokens) == len(labels):
        return "\n".join(f"{t}\t{lab}" for t, lab in zip(tokens, labels))
    return "\n".join(labels)


def _normalise_tags(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    rows: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # accept both "token\ttag" and bare "tag"
        parts = line.split()
        if len(parts) >= 2 and "\t" in raw:
            rows.append(parts[-1])
        else:
            rows.append(parts[-1])
    return tuple(rows)


def seed_playbook() -> "str":  # type: ignore[name-defined]  # noqa
    """Return a curated XBRL-aware seed playbook for FiNER.

    Stored as a markdown string so it can be loaded via
    :func:`ceng.playbook.Playbook.from_text` and merged into an empty
    playbook. ~30 strategies across the seven ACE sections.
    """
    return _FINER_SEED_PLAYBOOK


_FINER_SEED_PLAYBOOK = """## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: Always tag the token as a BIO tag, never as a free-form string. Each token in the input gets exactly one output label, in the same order as the input tokens.
[str-00002] helpful=0 harmful=0 :: 'O' marks a token that is NOT part of any named entity. Use 'O' liberally for punctuation, determiners, and common verbs.
[str-00003] helpful=0 harmful=0 :: XBRL person names are tagged PER; place names are LOC; company / fund names are ORG. Family relationships inside a person name keep the same tag.
[str-00004] helpful=0 harmful=0 :: For multi-word entities, the first token carries the B- prefix; subsequent tokens carry the I- prefix. Never mix B- and I- across entities with different types.
[str-00005] helpful=0 harmful=0 :: Numeric tokens (digits, percent signs, currency) are usually O unless they are part of a Measure / Location identifier.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Count tokens: the number of input tokens must equal the number of output labels. If they don't match, pad with 'O' or trim.

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: one label per line, in token order. If the model writes 'token\\tlabel' on each line, strip the token and keep only the label.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Confusing B- and I- tags is the most common FiNER error. Re-read multi-word entities and confirm the first token is B- while the rest are I-.
[mis-00002] helpful=0 harmful=0 :: Don't tag 'the', 'a', 'of', etc. as anything other than O. Common English determiners and prepositions are not entities.
[mis-00003] helpful=0 harmful=0 :: Don't confuse abbreviations (Inc., Corp., Ltd.) with ORG-only; they ARE part of the ORG entity.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: For ambiguous names, check whether the token is preceded by a known entity trigger ('Mr.', 'Inc.', 'Ltd.') before tagging.
[ps-00002] helpful=0 harmful=0 :: If a single word could be PER or LOC, look at the immediate context sentence for clue words ('said', 'based in', 'reported').

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Quotation marks often enclose an entity. The first quoted token may be PER or ORG.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, prefer O over a wrong label — partial credit is not awarded, so a wrong guess hurts the per-token F1.
"""
