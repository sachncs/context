"""FinanceBench: real SEC filings with questions, answers and evidence.

Source: PatronusAI/financebench on Hugging Face (CC BY-NC 4.0). The
documents are downloaded from their public issuer URLs on first use into a
local cache and are never redistributed with this package.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import pathlib
import urllib.request
from collections.abc import Sequence

from foveate import errors
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.tokenizers import base as tokenizer_base

DATASET_URL = (
    "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/"
    "financebench_merged.jsonl"
)
USER_AGENT = "Mozilla/5.0 (foveate benchmark; research use)"
TIMEOUT_SECONDS = 45


def cache_home() -> pathlib.Path:
    """Returns the cache root (`$FOVEATE_CACHE_HOME` or `~/.cache/foveate`)."""
    root = os.environ.get("FOVEATE_CACHE_HOME")
    return (
        pathlib.Path(root)
        if root
        else pathlib.Path.home() / ".cache" / "foveate"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """Gold evidence as shipped by FinanceBench.

    Attributes:
        text: Evidence passage.
        page_index: The dataset's 0-based page index.
        full_page: Full text of the evidence page, when provided.
    """

    text: str
    page_index: int
    full_page: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Question:
    """One FinanceBench item.

    Attributes:
        id: Dataset id.
        company: Company name.
        doc_name: Filing name, e.g. `3M_2018_10K`.
        doc_link: Public PDF URL.
        kind: `metrics-generated`, `domain-relevant` or `novel-generated`.
        question: The question.
        answer: Reference answer.
        justification: Reference rationale (may be empty).
        evidence: Gold evidence items.
    """

    id: str
    company: str
    doc_name: str
    doc_link: str
    kind: str
    question: str
    answer: str
    justification: str
    evidence: tuple[Evidence, ...]


def parse_evidence(raw: object) -> tuple[Evidence, ...]:
    """Parses the dataset's evidence field (list, or its string form)."""
    items = raw
    if isinstance(raw, str):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return ()
    found = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            found.append(
                Evidence(
                    str(item.get("evidence_text", "")),
                    int(item.get("evidence_page_num", 0)),
                    str(item.get("evidence_text_full_page", "")),
                )
            )
    return tuple(found)


def fetch(url: str) -> bytes:
    """Downloads `url` (follows redirects)."""
    if not url.startswith(("https://", "http://")):
        raise errors.ValidationError(f"refusing non-HTTP url: {url!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as reply:  # noqa: S310
            return bytes(reply.read())
    except OSError as exc:
        raise errors.ValidationError(f"cannot download {url}: {exc}") from exc


class FinanceBench:
    """Loads questions and documents, caching everything locally."""

    def __init__(self, home: pathlib.Path | None = None) -> None:
        """Creates the loader.

        Args:
            home: Cache root; defaults to `cache_home()`.
        """
        self.root = (home or cache_home()) / "financebench"
        self.root.mkdir(parents=True, exist_ok=True)

    def questions(self) -> list[Question]:
        """Returns all 150 questions (downloads the jsonl once)."""
        path = self.root / "financebench_merged.jsonl"
        if not path.exists():
            path.write_bytes(fetch(DATASET_URL))
        questions = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(
                Question(
                    id=row["financebench_id"],
                    company=row["company"],
                    doc_name=row["doc_name"],
                    doc_link=row["doc_link"],
                    kind=row["question_type"],
                    question=row["question"],
                    answer=str(row["answer"]),
                    justification=str(row.get("justification") or ""),
                    evidence=parse_evidence(row.get("evidence")),
                )
            )
        return questions

    def pdf(self, question: Question) -> pathlib.Path:
        """Returns the cached PDF path for the question's filing."""
        path = self.root / "pdf" / f"{question.doc_name}.pdf"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            data = fetch(question.doc_link)
            if not data.startswith(b"%PDF"):
                raise errors.ValidationError(
                    f"{question.doc_link} did not return a PDF"
                )
            path.write_bytes(data)
        return path

    def document(
        self,
        question: Question,
        tokenizer: tokenizer_base.Tokenizer | None = None,
    ) -> Document:
        """Returns the filing as a `Document` (parsed once, then cached)."""
        tokenizer = tokenizer or tokenizer_base.default_tokenizer()
        parsed = self.root / "parsed" / f"{question.doc_name}.json"
        if parsed.exists():
            texts = json.loads(parsed.read_text(encoding="utf-8"))
            pages = tuple(
                page_lib.Page(
                    i,
                    t,
                    tokenizer.count(t),
                    page_lib.detect_headings(t),
                )
                for i, t in enumerate(texts, start=1)
            )
            return Document(question.doc_name, pages, source=str(parsed))
        document = Document.load(
            self.pdf(question), doc_id=question.doc_name, tokenizer=tokenizer
        )
        parsed.parent.mkdir(exist_ok=True)
        parsed.write_text(
            json.dumps([p.text for p in document.pages]), encoding="utf-8"
        )
        return document


def stratified_sample(
    questions: Sequence[Question], n: int, seed: int = 0
) -> list[Question]:
    """Picks `n` questions, balanced over kinds and spread over filings.

    Deterministic for a given `seed`. Questions are round-robined across
    kinds, preferring filings not yet used so a small sample covers many
    documents.
    """
    import random

    rng = random.Random(seed)
    by_kind: dict[str, list[Question]] = {}
    for question in questions:
        by_kind.setdefault(question.kind, []).append(question)
    for pool in by_kind.values():
        rng.shuffle(pool)
    chosen: list[Question] = []
    used_docs: set[str] = set()
    while len(chosen) < min(n, len(questions)):
        progressed = False
        for kind in sorted(by_kind):
            pool = by_kind[kind]
            pick = next((q for q in pool if q.doc_name not in used_docs), None)
            pick = pick or (pool[0] if pool else None)
            if pick is None:
                continue
            pool.remove(pick)
            chosen.append(pick)
            used_docs.add(pick.doc_name)
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return chosen
