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

from foveate import errors
from foveate import messages as messages_lib
from foveate.bench import scoring
from foveate.bench.longdoc import dataset
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
        pages: For the document task, the page texts (to feed `foveate`).
        atoms: Facts that must survive compression (for retention metrics).
        golds: All acceptable answers (for exact match and F1); defaults to
            `expected`.
    """

    messages: tuple[messages_lib.Message, ...]
    question: str
    expected: str
    pages: tuple[str, ...] = ()
    atoms: tuple[scoring.Atom, ...] = ()
    golds: tuple[str, ...] = ()

    @property
    def answers(self) -> tuple[str, ...]:
        """Returns the acceptable answers."""
        return self.golds or (self.expected,)

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


@Task.registry.register("atoms")
class Atoms(Task):
    """A long chat that states several facts once each; all should survive."""

    name = "atoms"
    TEMPLATES = (
        ("deployment region", "Our deployment region is {v}.", 3),
        ("budget", "The approved budget is {v} thousand dollars.", 2),
        ("owner", "The project owner is {v}.", 3),
        ("deadline", "The hard deadline is {v}.", 2),
        ("database port", "The database listens on port {v}.", 1),
        ("rollback policy", "The rollback policy is {v}.", 1),
    )
    OWNERS = ("Priya Raman", "Tomas Weber", "Ayesha Khan", "Lucas Moreau")
    DEADLINES = ("March 14", "June 2", "September 30", "November 21")
    POLICIES = ("automatic", "manual approval", "blue-green", "canary")

    def values(self, rng: random.Random) -> list[str]:
        """Returns one invented value per template."""
        return [
            rng.choice(REGIONS),
            str(rng.randrange(120, 980)),
            rng.choice(self.OWNERS),
            rng.choice(self.DEADLINES),
            str(rng.randrange(5000, 9000)),
            rng.choice(self.POLICIES),
        ]

    def build(self, seed: int) -> Sample:
        rng = random.Random(f"atoms:{seed}")
        values = self.values(rng)
        atoms = tuple(
            scoring.Atom(template[0], value, template[2])
            for template, value in zip(self.TEMPLATES, values, strict=True)
        )
        turns = max(24, self.tokens // 90)
        slots = sorted(
            rng.sample(range(turns // 6, turns * 3 // 4), len(atoms))
        )
        planted = dict(zip(slots, range(len(atoms)), strict=True))
        out: list[messages_lib.Message] = []
        for turn in range(turns):
            if turn in planted:
                sentence = self.TEMPLATES[planted[turn]][1]
                text = sentence.format(v=values[planted[turn]])
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
        asked = atoms[0]
        return Sample(
            tuple(out),
            f"What is the {asked.key}?",
            asked.value,
            atoms=atoms,
        )


HOTPOT_ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa"
    "&config=distractor&split=validation&offset={offset}&length=1"
)


@Task.registry.register("qa")
class HotpotQA(Task):
    """HotpotQA (distractor setting): answer from ten paragraphs, two relevant.

    Public multi-hop question answering, the kind of data context-compression
    papers evaluate on (exact match and F1 at 4x and 8x). Rows are downloaded
    one at a time and cached.
    """

    name = "qa"

    def fetch(self, index: int) -> dict[str, object]:
        """Returns the validation row `index` (cached on disk)."""
        path = dataset.cache_home() / "hotpotqa" / f"validation_{index}.json"
        if not path.exists():
            raw = json.loads(
                dataset.fetch(HOTPOT_ROWS.format(offset=index * 37))
            )
            try:
                row = raw["rows"][0]["row"]
            except (KeyError, IndexError, TypeError) as exc:
                raise errors.ValidationError(
                    f"unexpected HotpotQA reply: {exc}"
                ) from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row), encoding="utf-8")
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise errors.ValidationError("HotpotQA row is not an object")
        return row

    def build(self, seed: int) -> Sample:
        row = self.fetch(seed)
        context = row["context"]
        titles = context["title"]  # type: ignore[index]
        sentences = context["sentences"]  # type: ignore[index]
        paragraphs = [
            f"{title}: {''.join(parts).strip()}"
            for title, parts in zip(titles, sentences, strict=True)
        ]
        answer = str(row["answer"])
        return Sample(
            (Message(Role.USER, "\n\n".join(paragraphs)),),
            str(row["question"]),
            answer,
            golds=(answer,),
        )


Message = messages_lib.Message
