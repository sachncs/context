"""The `Document` value type."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from foveate import errors
from foveate import messages as messages_lib
from foveate.documents import page as page_lib
from foveate.tokenizers import base as tokenizer_base


@dataclasses.dataclass(frozen=True, slots=True)
class Document:
    """An immutable, page-addressable document.

    Attributes:
        id: Stable identifier used in citations (defaults to the file stem).
        pages: Pages in reading order with unique, increasing numbers.
        title: Human title (may be empty).
        source: Where it was loaded from (path or URL; informational).
        metadata: String annotations (loader name, empty-page count, ...).
    """

    id: str
    pages: tuple[page_lib.Page, ...]
    title: str = ""
    source: str = ""
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise errors.ValidationError("a document needs an id")
        numbers = [p.number for p in self.pages]
        if numbers != sorted(set(numbers)):
            raise errors.ValidationError(
                "page numbers must be unique and in increasing order"
            )
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @classmethod
    def load(
        cls,
        source: str | pathlib.Path | bytes,
        *,
        format: str | None = None,
        doc_id: str | None = None,
        tokenizer: tokenizer_base.Tokenizer | None = None,
        **options: object,
    ) -> Document:
        """Loads a document from a path or bytes.

        Args:
            source: File path, or raw bytes (then `format` is required).
            format: Loader name (`pdf`, `text`, `markdown`, `html`, `docx`);
                inferred from the file suffix when omitted.
            doc_id: Citation id; defaults to the file stem.
            tokenizer: Counts page tokens; defaults to the best available.
            **options: Loader options (for example `page_tokens=500` for
                formats without real pages).

        Raises:
            ValidationError: For unreadable or unsupported input.
            ConfigError: If an optional dependency is missing.
        """
        from foveate.documents import loaders

        return loaders.load(
            source,
            format=format,
            doc_id=doc_id,
            tokenizer=tokenizer,
            options=options,
        )

    @property
    def token_count(self) -> int:
        """Returns the total tokens over all pages."""
        return sum(p.tokens for p in self.pages)

    @property
    def numbers(self) -> list[int]:
        """Returns the page numbers present, in order."""
        return [p.number for p in self.pages]

    def page(self, number: int) -> page_lib.Page:
        """Returns one page by number.

        Raises:
            ValidationError: If the document has no such page.
        """
        for candidate in self.pages:
            if candidate.number == number:
                return candidate
        raise errors.ValidationError(f"{self.id} has no page {number}")

    def select(self, spec: str | int | range) -> Document:
        """Returns a document holding only the requested pages.

        Args:
            spec: `"10-14,40"`, a page number, or a `range`.

        Raises:
            ValidationError: For malformed specs or missing pages.
        """
        wanted = set(page_lib.parse_pages(spec, self.numbers))
        return dataclasses.replace(
            self, pages=tuple(p for p in self.pages if p.number in wanted)
        )

    def around(self, number: int, radius: int = 1) -> Document:
        """Returns a page and its `radius` neighbours on each side.

        Raises:
            ValidationError: If `number` is not a page of the document.
        """
        self.page(number)
        low, high = number - radius, number + radius
        return dataclasses.replace(
            self,
            pages=tuple(p for p in self.pages if low <= p.number <= high),
        )

    def outline(self) -> list[page_lib.Heading]:
        """Returns detected headings with their pages, in reading order."""
        return [
            page_lib.Heading(title, p.number)
            for p in self.pages
            for title in p.headings
        ]

    def text(self, markers: bool = False) -> str:
        """Returns the document text.

        Args:
            markers: Prefix each page with `[doc_id p.N]` so a model can cite.
        """
        parts = []
        for p in self.pages:
            parts.append(
                f"[{self.id} p.{p.number}]\n{p.text}" if markers else p.text
            )
        return "\n\n".join(parts)

    def to_messages(
        self, role: messages_lib.Role = messages_lib.Role.USER
    ) -> list[messages_lib.Message]:
        """Returns one marker-prefixed message per page.

        One message per page lets compression strategies treat pages
        independently while every page stays citable.
        """
        return [
            messages_lib.Message(role, f"[{self.id} p.{p.number}]\n{p.text}")
            for p in self.pages
            if p.text.strip()
        ]


def unique_ids(documents: Sequence[Document]) -> None:
    """Raises `ValidationError` if two documents share an id."""
    seen: set[str] = set()
    for document in documents:
        if document.id in seen:
            raise errors.ValidationError(
                f"duplicate document id {document.id!r}"
            )
        seen.add(document.id)
