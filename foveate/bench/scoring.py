"""Answer and retention metrics shared by the benchmarks.

* Exact match and token F1 follow the SQuAD protocol used by MRQA, HotpotQA
  and the compression papers that evaluate on them.
* Atom metrics follow the commitment-preservation idea of context codecs: plant
  a few typed facts ("atoms"), compress, and measure how many survive,
  weighted by importance, without asking a model to judge.
"""

from __future__ import annotations

import collections
import dataclasses
import re
import string
from collections.abc import Sequence

from foveate.tokenizers import base as tokenizer_base

ARTICLES = re.compile(r"\b(a|an|the)\b")
KEPT = "kept"
OMITTED = "omitted"
MUTATED = "mutated"


def normalize_answer(text: str) -> str:
    """Lower-cases and strips punctuation, articles and extra spaces."""
    lowered = text.casefold()
    stripped = "".join(c for c in lowered if c not in string.punctuation)
    return " ".join(ARTICLES.sub(" ", stripped).split())


def exact_match(prediction: str, gold: str) -> bool:
    """Returns whether the normalised strings are equal."""
    return normalize_answer(prediction) == normalize_answer(gold)


def contains(prediction: str, gold: str) -> bool:
    """Returns whether the normalised prediction contains the gold answer."""
    wanted = normalize_answer(gold)
    return bool(wanted) and wanted in normalize_answer(prediction)


def token_f1(prediction: str, gold: str) -> float:
    """Returns the token-overlap F1 of two answers (0 to 1)."""
    left = normalize_answer(prediction).split()
    right = normalize_answer(gold).split()
    if not left or not right:
        return float(left == right)
    common = collections.Counter(left) & collections.Counter(right)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(left)
    recall = same / len(right)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, golds: Sequence[str]) -> float:
    """Returns the best token F1 over several acceptable answers."""
    return max((token_f1(prediction, g) for g in golds), default=0.0)


@dataclasses.dataclass(frozen=True, slots=True)
class Atom:
    """One fact that must survive compression.

    Attributes:
        key: What the fact is about (for example `deployment region`).
        value: The exact value that must remain recoverable.
        weight: Importance; critical atoms weigh more.
    """

    key: str
    value: str
    weight: int = 1


def status(atom: Atom, text: str) -> str:
    """Classifies an atom in `text` as kept, mutated or omitted.

    `mutated` means the topic survived but its value did not (a changed or
    weakened commitment); `omitted` means neither did.
    """
    haystack = normalize_answer(text)
    if normalize_answer(atom.value) in haystack:
        return KEPT
    if normalize_answer(atom.key) in haystack:
        return MUTATED
    return OMITTED


def critical_atom_recall(atoms: Sequence[Atom], text: str) -> float:
    """Returns the share of atoms whose value is still present."""
    if not atoms:
        return 1.0
    return sum(status(a, text) == KEPT for a in atoms) / len(atoms)


def weighted_atom_recall(atoms: Sequence[Atom], text: str) -> float:
    """Returns recall with each atom counted by its weight."""
    total = sum(a.weight for a in atoms)
    if total == 0:
        return 1.0
    kept = sum(a.weight for a in atoms if status(a, text) == KEPT)
    return kept / total


def commitment_density(
    atoms: Sequence[Atom],
    text: str,
    tokenizer: tokenizer_base.Tokenizer,
) -> float:
    """Returns kept atoms per 1,000 tokens of `text`."""
    tokens = max(1, tokenizer.count(text))
    kept = sum(status(a, text) == KEPT for a in atoms)
    return 1000.0 * kept / tokens


def taxonomy(atoms: Sequence[Atom], text: str) -> dict[str, int]:
    """Counts atoms by outcome: kept, mutated, omitted."""
    counts = {KEPT: 0, MUTATED: 0, OMITTED: 0}
    for atom in atoms:
        counts[status(atom, text)] += 1
    return counts
