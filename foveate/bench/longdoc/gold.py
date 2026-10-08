"""Gold sets for long-document evaluation, built on FinanceBench.

Four kinds of item, each with a known right behaviour:

* `answerable`: a real question with its reference answer and the **pages
  that contain the evidence** (located by matching the evidence text in the
  parsed PDF, not by trusting the dataset's page index).
* `unanswerable`: the same question asked of a filing that provably lacks the
  answer (different company, answer absent); the right behaviour is to
  abstain.
* `paraphrase`: meaning-preserving rewrites of an answerable question, used
  to measure how stable answers are.
* `needle`: a synthetic sentence inserted at a chosen depth of a real filing,
  with a random answer no model can have memorised (lost-in-the-middle).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from foveate import errors
from foveate.bench.longdoc import dataset
from foveate.documents import Document
from foveate.grounding import quotes

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"
PARAPHRASE = "paraphrase"
NEEDLE = "needle"
KINDS = (ANSWERABLE, UNANSWERABLE, PARAPHRASE, NEEDLE)
EVIDENCE_OVERLAP = 0.7
NUMBER = re.compile(r"\d[\d,]*\.?\d*")
CODE_SYLLABLES = ("ka", "lo", "mi", "ra", "su", "te", "vo", "zu", "ne", "pi")
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class GoldItem:
    """One evaluation item.

    Attributes:
        id: Unique id (`<kind>:<source id>[:<variant>]`).
        kind: One of `KINDS`.
        doc_name: Filing the question is asked against.
        question: The question text.
        expected: Reference answer; empty for unanswerable items.
        gold_pages: 1-based pages containing the evidence (empty if none).
        meta: Kind-specific details (source id, depth, needle sentence, ...).
    """

    id: str
    kind: str
    doc_name: str
    question: str
    expected: str
    gold_pages: tuple[int, ...] = ()
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise errors.ValidationError(f"unknown gold kind {self.kind!r}")
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def to_json(self) -> str:
        """Serialises to one JSON line."""
        data = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }
        data["meta"] = dict(self.meta)
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> GoldItem:
        """Parses one JSON line.

        Raises:
            ValidationError: If the line is malformed.
        """
        try:
            data = json.loads(line)
            return cls(
                id=data["id"],
                kind=data["kind"],
                doc_name=data["doc_name"],
                question=data["question"],
                expected=data["expected"],
                gold_pages=tuple(data["gold_pages"]),
                meta=data.get("meta", {}),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise errors.ValidationError(f"bad gold line: {exc}") from exc


def write_items(path: pathlib.Path, items: Iterable[GoldItem]) -> None:
    """Writes items as sorted JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(item.to_json() for item in items)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_items(path: pathlib.Path) -> list[GoldItem]:
    """Reads items from a JSON-lines file."""
    return [
        GoldItem.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def locate_pages(
    document: Document, evidence_text: str, hint_index: int | None = None
) -> tuple[int, ...]:
    """Finds the pages that contain an evidence passage.

    A page matches when at least `EVIDENCE_OVERLAP` of the passage's words
    occur on it. The dataset's 0-based page hint is tried first (and its
    neighbours), then the whole document.

    Returns:
        Matching page numbers (best first); empty if none matches.
    """
    wanted = set(quotes.normalise(evidence_text).split())
    if len(wanted) < 3:
        return ()
    order = list(document.numbers)
    if hint_index is not None:
        near = [
            n
            for n in (hint_index + 1, hint_index, hint_index + 2)
            if n in order
        ]
        order = near + [n for n in order if n not in near]
    scored = []
    for number in order:
        words = set(quotes.normalise(document.page(number).text).split())
        overlap = len(wanted & words) / len(wanted)
        if overlap >= EVIDENCE_OVERLAP:
            scored.append((overlap, number))
    scored.sort(key=lambda item: -item[0])
    return tuple(number for value, number in scored[:3])


def answerable_item(
    question: dataset.Question, document: Document
) -> GoldItem | None:
    """Builds an answerable item, or None if the evidence cannot be located."""
    pages: list[int] = []
    for evidence in question.evidence:
        found = locate_pages(document, evidence.text, evidence.page_index)
        pages.extend(p for p in found[:1] if p not in pages)
    if not pages:
        return None
    numbers = key_numbers(question.answer)
    on_page = "na"
    if numbers:
        texts = " ".join(document.page(p).text for p in pages).replace(",", "")
        on_page = "1" if any(trim_zeros(n) in texts for n in numbers) else "0"
    return GoldItem(
        id=f"{ANSWERABLE}:{question.id}",
        kind=ANSWERABLE,
        doc_name=question.doc_name,
        question=question.question,
        expected=question.answer,
        gold_pages=tuple(sorted(pages)),
        meta={
            "source": question.id,
            "category": question.kind,
            "company": question.company,
            "number_on_page": on_page,
        },
    )


def trim_zeros(number: str) -> str:
    """Drops a trailing all-zero fraction ("1577.00" -> "1577")."""
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def key_numbers(answer: str) -> list[str]:
    """Returns the answer's significant numbers, normalised."""
    found = []
    for raw in NUMBER.findall(answer):
        digits = raw.replace(",", "").rstrip(".")
        if len(digits.replace(".", "")) >= 3:
            found.append(digits)
    return found


def lacks_answer(wrong: Document, question: dataset.Question) -> bool:
    """Returns whether `wrong` provably cannot answer `question`.

    True when neither the company name nor any key number of the reference
    answer appears in the document text.
    """
    text = quotes.normalise(wrong.text())
    company = quotes.normalise(question.company)
    if company and company in text:
        return False
    return not any(
        quotes.normalise(n) in text for n in key_numbers(question.answer)
    )


def unanswerable_item(
    question: dataset.Question, wrong_doc: Document
) -> GoldItem | None:
    """Pairs `question` with a filing that lacks the answer, if it does."""
    if wrong_doc.id == question.doc_name or not lacks_answer(
        wrong_doc, question
    ):
        return None
    return GoldItem(
        id=f"{UNANSWERABLE}:{question.id}:{wrong_doc.id}",
        kind=UNANSWERABLE,
        doc_name=wrong_doc.id,
        question=question.question,
        expected="",
        gold_pages=(),
        meta={"source": question.id, "original_doc": question.doc_name},
    )


def needle_items(
    document: Document, seed: int, depths: Sequence[float] = DEPTHS
) -> list[GoldItem]:
    """Builds one needle item per depth for `document`.

    Each needle is a sentence with a random invented codename, inserted at
    the given relative depth into the page found there.
    """
    rng = random.Random(f"{seed}:{document.id}")
    items = []
    numbers = document.numbers
    for depth in depths:
        page_number = numbers[min(len(numbers) - 1, int(depth * len(numbers)))]
        code = "".join(rng.choice(CODE_SYLLABLES) for count in range(4)).title()
        topic = rng.choice(
            ("vault audit", "records migration", "supplier review")
        )
        sentence = (
            f"For the record, the internal codename of the {topic} is {code}-"
            f"{rng.randint(100, 999)}."
        )
        items.append(
            GoldItem(
                id=f"{NEEDLE}:{document.id}:{int(depth * 100)}",
                kind=NEEDLE,
                doc_name=document.id,
                question=f"What is the internal codename of the {topic}?",
                expected=sentence.rsplit(" ", 1)[1].rstrip("."),
                gold_pages=(page_number,),
                meta={"depth": str(depth), "sentence": sentence},
            )
        )
    return items


def with_needle(document: Document, item: GoldItem) -> Document:
    """Returns `document` with the item's needle sentence inserted."""
    from foveate.documents import page as page_lib

    sentence = item.meta["sentence"]
    target = item.gold_pages[0]
    pages = []
    for page in document.pages:
        if page.number == target:
            middle = len(page.text) // 2
            cut = page.text.find("\n", middle)
            cut = len(page.text) if cut == -1 else cut
            text = page.text[:cut] + "\n" + sentence + "\n" + page.text[cut:]
            pages.append(
                page_lib.Page(
                    page.number, text, page.tokens + 40, page.headings
                )
            )
        else:
            pages.append(page)
    return dataclasses.replace(document, pages=tuple(pages))
