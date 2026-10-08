"""Needle-in-a-haystack cases: literal, non-literal (NoLiMa style), multi.

A `Case` says what to hide, where, and what the right answer is. Building a
case needs no model and no download; the haystack is supplied separately.
"""

from __future__ import annotations

import dataclasses
import random
import string
from collections.abc import Mapping, Sequence
from typing import Any

from foveate import errors

LITERAL = "literal"
NONLITERAL = "nonliteral"
MULTI = "multi"
FAMILIES = (LITERAL, NONLITERAL, MULTI)
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
LENGTHS = (8_000, 16_000, 32_000, 64_000, 128_000)
TOPICS = (
    "vault audit",
    "records migration",
    "supplier review",
    "fire drill",
    "pilot programme",
    "archive census",
    "tender round",
    "safety briefing",
)
INSTRUCTION = "Answer using only the document. Reply with the answer only."


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """One needle test.

    Attributes:
        id: Stable identifier.
        family: `literal`, `nonliteral` or `multi`.
        length: Target haystack tokens.
        depths: Relative depth of each needle (one per sentence).
        sentences: The needle sentences to hide.
        question: What to ask.
        expected: The answer; several required parts are joined by `|`.
        seed: Seed used for the haystack and the invented values.
    """

    id: str
    family: str
    length: int
    depths: tuple[float, ...]
    sentences: tuple[str, ...]
    question: str
    expected: str
    seed: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise errors.ConfigError(f"unknown needle family {self.family!r}")
        if len(self.depths) != len(self.sentences) or not self.sentences:
            raise errors.ConfigError("one depth per needle sentence")

    @property
    def depth(self) -> float:
        """Returns the depth of the first needle (the sweep coordinate)."""
        return self.depths[0]

    def correct(self, answer: str) -> bool:
        """Returns whether `answer` contains every required part."""
        text = answer.casefold()
        return all(part.casefold() in text for part in self.expected.split("|"))


def code(rng: random.Random) -> str:
    """Returns an invented, unguessable code such as `QX7-482`."""
    letters = "".join(rng.choice(string.ascii_uppercase) for place in range(2))
    return f"{letters}{rng.randint(1, 9)}-{rng.randint(100, 999)}"


def literal(length: int, depth: float, seed: int) -> Case:
    """A needle that shares its key words with the question (classic NIAH)."""
    rng = random.Random(f"literal:{length}:{depth}:{seed}")
    topic = rng.choice(TOPICS)
    value = code(rng)
    return Case(
        id=f"{LITERAL}:{length}:{int(depth * 100)}:{seed}",
        family=LITERAL,
        length=length,
        depths=(depth,),
        sentences=(
            f"For the record, the internal codename of the {topic} is {value}.",
        ),
        question=f"What is the internal codename of the {topic}?",
        expected=value,
        seed=seed,
    )


def multi(length: int, count: int, seed: int, depth: float = 0.0) -> Case:
    """Several keyed needles spread over the haystack; all must be returned.

    `depth` is the position of the first needle; the others are spread evenly
    over the rest so the sweep coordinate stays meaningful.
    """
    if count < 2:
        raise errors.ConfigError("multi needs at least two needles")
    rng = random.Random(f"multi:{length}:{count}:{depth}:{seed}")
    topics = rng.sample(TOPICS, count)
    values = [code(rng) for place in range(count)]
    span = 1.0 - depth
    depths = tuple(min(1.0, depth + span * i / count) for i in range(count))
    sentences = tuple(
        f"For the record, the internal codename of the {t} is {v}."
        for t, v in zip(topics, values, strict=True)
    )
    names = ", ".join(f"the {t}" for t in topics)
    return Case(
        id=f"{MULTI}:{length}:{count}:{int(depth * 100)}:{seed}",
        family=MULTI,
        length=length,
        depths=depths,
        sentences=sentences,
        question=f"List the internal codenames of: {names}.",
        expected="|".join(values),
        seed=seed,
    )


def fill(template: str, args: Sequence[str], character: str) -> str:
    """Fills a NoLiMa template: `{CHAR}` and `{1}`, `{2}`, `{3}`."""
    out = template.replace("{CHAR}", character)
    for number, value in enumerate(args, start=1):
        out = out.replace(f"{{{number}}}", value)
    return out


def nonliteral(
    needle_set: Sequence[Mapping[str, Any]],
    index: int,
    test: str,
    length: int,
    depth: float,
    seed: int,
    hop: str = "onehop",
) -> Case:
    """A NoLiMa case: the question shares no words with the needle.

    Args:
        needle_set: The parsed `needle_set.json` of NoLiMa.
        index: Which needle template to use.
        test: Which test inside it (for example `"T17_C02"`).
        length: Target haystack tokens.
        depth: Relative depth.
        seed: Chooses the character name.
        hop: `onehop` or `twohop` question.

    Raises:
        ConfigError: For an unknown template, test or hop.
    """
    try:
        entry = needle_set[index]
        args = entry["tests"][test]["input_args"]
        question_template = entry["questions"][hop]
        characters = entry["character_set"]
        template = str(entry["needle"])
    except (IndexError, KeyError, TypeError) as exc:
        raise errors.ConfigError(f"bad NoLiMa selection: {exc}") from exc
    rng = random.Random(f"nolima:{index}:{test}:{seed}")
    character = str(rng.choice(list(characters)))
    values = [str(a) for a in args]
    return Case(
        id=(
            f"{NONLITERAL}:{entry['id']}:{test}:{hop}:{length}:"
            f"{int(depth * 100)}:{seed}"
        ),
        family=NONLITERAL,
        length=length,
        depths=(depth,),
        sentences=(fill(template, values, character),),
        question=fill(str(question_template), values, character),
        expected=character,
        seed=seed,
    )


def grid(
    lengths: Sequence[int] = LENGTHS,
    depths: Sequence[float] = DEPTHS,
    seed: int = 0,
) -> list[Case]:
    """Returns the literal-needle grid: every length at every depth."""
    return [literal(n, d, seed) for n in lengths for d in depths]
