"""Chunks: retrieval units that remember their page."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import ClassVar

from foveate import errors
from foveate.documents import document as document_lib
from foveate.internals import registry
from foveate.partition import base as partition_base
from foveate.tokenizers import base as tokenizer_base


@dataclasses.dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of one page.

    Attributes:
        doc_id: Owning document id.
        page: 1-based page number (the citation target).
        index: Position of the chunk within its page.
        text: Chunk text.
        tokens: Token count of `text`.
        heading: Nearest heading at or before this chunk's page.
    """

    doc_id: str
    page: int
    index: int
    text: str
    tokens: int
    heading: str = ""

    @property
    def id(self) -> str:
        """Returns a stable `doc:page:index` identifier."""
        return f"{self.doc_id}:{self.page}:{self.index}"

    @property
    def searchable(self) -> str:
        """Returns the text indexed for retrieval (heading plus body)."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


class Chunker(abc.ABC):
    """Splits documents into chunks."""

    registry: ClassVar[registry.Registry[type[Chunker]]] = registry.Registry(
        "chunker"
    )

    @abc.abstractmethod
    def chunk(
        self,
        document: document_lib.Document,
        tokenizer: tokenizer_base.Tokenizer,
    ) -> list[Chunk]:
        """Returns the document's chunks in reading order."""


@Chunker.registry.register("page")
class PageChunker(Chunker):
    """One chunk per page, subdivided when a page exceeds `max_tokens`.

    Attributes:
        max_tokens: Ceiling per chunk.
    """

    def __init__(self, max_tokens: int = 256) -> None:
        """Creates the chunker.

        Raises:
            ConfigError: If `max_tokens` is not positive.
        """
        if max_tokens < 1:
            raise errors.ConfigError("max_tokens must be >= 1")
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: document_lib.Document,
        tokenizer: tokenizer_base.Tokenizer,
    ) -> list[Chunk]:
        splitter = partition_base.RecursivePartitioner(self.max_tokens)
        chunks: list[Chunk] = []
        heading = ""
        for page in document.pages:
            if page.headings:
                heading = page.headings[0]
            if not page.text.strip():
                continue
            for part in splitter.split(page.text, tokenizer):
                if not part.text.strip():
                    continue
                chunks.append(
                    Chunk(
                        doc_id=document.id,
                        page=page.number,
                        index=part.index,
                        text=part.text,
                        tokens=part.tokens,
                        heading=heading,
                    )
                )
        return chunks


def chunk_all(
    documents: Sequence[document_lib.Document],
    tokenizer: tokenizer_base.Tokenizer,
    chunker: Chunker | None = None,
) -> list[Chunk]:
    """Chunks several documents with one chunker (default `PageChunker`)."""
    document_lib.unique_ids(documents)
    use = chunker or PageChunker()
    return [c for d in documents for c in use.chunk(d, tokenizer)]
