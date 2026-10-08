"""Compression test material: a fact is planted, the question needs it.

Each task builds a `Sample` from a seed with no model and no download. The
planted fact sits where naive compression tends to lose it (the middle).
"""

from __future__ import annotations

import abc
import dataclasses
import json
import random
from typing import ClassVar

from foveate import messages as messages_lib
from foveate.bench.needle import cases, haystack
from foveate.internals import registry
from foveate.tokenizers import base as tokenizer_base

Role = messages_lib.Role
REGIONS = ("eu-west-1", "us-east-2", "ap-south-1", "eu-north-1", "sa-east-1")
TOPICS = (
    "the roadmap",
    "the lunch order",
    "the printer queue",
    "the holiday rota",
    "the travel policy",
    "the badge system",
    "the parking permit",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    """What a compressor sees and what the reader is then asked.

    Attributes:
        messages: The context to compress.
        question: Asked after compression.
        expected: Text the right answer contains.
        query: Hint for query-aware methods (the question).
        pages: For the document task, the page texts (to feed `foveate`).
    """

    messages: tuple[messages_lib.Message, ...]
    question: str
    expected: str
    pages: tuple[str, ...] = ()

    @property
    def query(self) -> str:
        """Returns the question, used by query-aware methods."""
        return self.question


class Task(abc.ABC):
    """A family of samples with one planted fact each."""

    registry: ClassVar[registry.Registry[type[Task]]] = registry.Registry(
        "compression task"
    )
    name: ClassVar[str] = ""

    def __init__(
        self, tokenizer: tokenizer_base.Tokenizer, tokens: int = 6_000
    ) -> None:
        """Creates the task.

        Args:
            tokenizer: Counts tokens.
            tokens: Rough size of the uncompressed context.
        """
        self.tokenizer = tokenizer
        self.tokens = tokens

    @abc.abstractmethod
    def build(self, seed: int) -> Sample:
        """Returns the sample for `seed`."""


@Task.registry.register("history")
class History(Task):
    """A long chat in which the user once states a setting; asked at the end."""

    name = "history"

    def build(self, seed: int) -> Sample:
        rng = random.Random(f"history:{seed}")
        region = rng.choice(REGIONS)
        turns = max(12, self.tokens // 90)
        planted = rng.randrange(turns // 5, turns * 3 // 5)
        out: list[messages_lib.Message] = []
        for turn in range(turns):
            if turn == planted:
                text = f"By the way, our deployment region is {region}."
            else:
                text = (
                    " ".join(haystack.prose_sentences(2, rng))
                    + f" (regarding {rng.choice(TOPICS)})"
                )
            out.append(Message(Role.USER, text))
            out.append(
                Message(
                    Role.ASSISTANT,
                    " ".join(haystack.prose_sentences(3, rng)),
                )
            )
        return Sample(
            tuple(out), "Which deployment region did the user mention?", region
        )


@Task.registry.register("tool")
class Tool(Task):
    """One large JSON tool result; the question needs a single field."""

    name = "tool"

    def build(self, seed: int) -> Sample:
        rng = random.Random(f"tool:{seed}")
        count = max(20, self.tokens // 45)
        target = rng.randrange(count // 4, count * 3 // 4)
        rows = []
        for index in range(count):
            rows.append(
                {
                    "order_id": f"OR-{1000 + index}",
                    "customer": " ".join(haystack.prose_sentences(1, rng))[:60],
                    "status": rng.choice(("shipped", "pending", "packed")),
                    "ref": cases.code(rng),
                }
            )
        order = rows[target]["order_id"]
        status = "delayed-by-customs"
        rows[target]["status"] = status
        payload = json.dumps({"results": rows, "total": count})
        return Sample(
            (
                Message(Role.USER, "Check the orders."),
                Message(Role.TOOL, payload, name="list_orders"),
            ),
            f"What is the status of order {order}?",
            status,
        )


@Task.registry.register("document")
class Document(Task):
    """A long document with one needle sentence at 50% depth."""

    name = "document"

    def build(self, seed: int) -> Sample:
        case = cases.literal(self.tokens, 0.5, seed)
        pages = haystack.prose_pages(case.length, self.tokenizer, seed)
        pages = haystack.insert(pages, case.sentences[0], case.depth)
        return Sample(
            (Message(Role.USER, "\n\n".join(pages)),),
            case.question,
            case.expected,
            tuple(pages),
        )


Message = messages_lib.Message
