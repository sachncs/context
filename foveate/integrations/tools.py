"""Framework-neutral document tools for agents.

An agent that can call `read_pages` and `search_document` pulls only the pages
it needs instead of receiving the whole document. The framework modules wrap
`DocumentTools.functions()` in their own tool types.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from foveate import errors
from foveate.documents import Document, chunking
from foveate.documents import document as document_lib
from foveate.internals import runner
from foveate.selection import base as selection_base
from foveate.selection import bm25
from foveate.tokenizers import HeuristicTokenizer
from foveate.tokenizers import base as tokenizer_base

MAX_OUTLINE_LINES = 200


@dataclasses.dataclass(frozen=True)
class DocumentTools:
    """Page-addressable access to documents for an agent.

    Attributes:
        documents: The documents the agent may read (unique ids).
        max_tokens: Largest tool reply, in tokens; longer replies are cut and
            say so, so the agent can ask for fewer pages.
        tokenizer: Counts tokens for `max_tokens`.
        retriever: Search index; built (BM25) from `documents` when None.
        index: The retriever in use (set on construction).
    """

    documents: Sequence[Document]
    max_tokens: int = 4000
    tokenizer: tokenizer_base.Tokenizer = dataclasses.field(
        default_factory=HeuristicTokenizer
    )
    retriever: selection_base.Retriever | None = None
    index: selection_base.Retriever = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.max_tokens < 100:
            raise errors.ConfigError("max_tokens must be at least 100")
        document_lib.unique_ids(self.documents)
        index = self.retriever
        if index is None:
            chunks = chunking.chunk_all(self.documents, self.tokenizer)
            index = bm25.BM25Retriever(chunks)
        object.__setattr__(self, "index", index)

    def document(self, doc_id: str) -> Document:
        """Returns the document `doc_id` (the only one when empty).

        Raises:
            ValidationError: If the id is unknown or ambiguous.
        """
        if not doc_id and len(self.documents) == 1:
            return self.documents[0]
        for candidate in self.documents:
            if candidate.id == doc_id:
                return candidate
        known = ", ".join(d.id for d in self.documents)
        raise errors.ValidationError(
            f"unknown document {doc_id!r}; available: {known}"
        )

    def fit(self, text: str) -> str:
        """Cuts `text` to `max_tokens`, saying so when it is cut."""
        if self.tokenizer.count(text) <= self.max_tokens:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.tokenizer.count(text[:middle]) <= self.max_tokens:
                low = middle
            else:
                high = middle - 1
        return text[:low] + "\n[truncated: reply too long; request fewer pages]"

    def read(self, pages: str, doc_id: str = "") -> str:
        """Returns the text of the requested pages, each marked for citing."""
        try:
            chosen = self.document(doc_id).select(pages)
        except errors.FoveateError as exc:
            return f"error: {exc}"
        return self.fit(chosen.text(markers=True))

    def search(self, query: str, k: int = 5, doc_id: str = "") -> str:
        """Returns the `k` best-matching pages with a snippet each."""
        hits = runner.run_sync(self.index.search(query, k * 4))
        if doc_id:
            hits = [h for h in hits if h.chunk.doc_id == doc_id]
        lines = []
        for (owner, page), score in selection_base.ranked_pages(hits)[:k]:
            best = max(
                (
                    h
                    for h in hits
                    if (h.chunk.doc_id, h.chunk.page) == (owner, page)
                ),
                key=lambda h: h.score,
            )
            snippet = " ".join(best.chunk.text.split())[:300]
            lines.append(f"[{owner} p.{page}] (score {score:.2f}) {snippet}")
        return self.fit("\n".join(lines) or "no matching pages")

    def outline(self, doc_id: str = "") -> str:
        """Returns the headings with page numbers, and the page count."""
        try:
            doc = self.document(doc_id)
        except errors.FoveateError as exc:
            return f"error: {exc}"
        lines = [f"{doc.id}: {len(doc.pages)} pages, {doc.token_count} tokens"]
        lines.extend(
            f"p.{h.page} {h.title}" for h in doc.outline()[:MAX_OUTLINE_LINES]
        )
        return self.fit("\n".join(lines))

    def functions(self) -> list[Callable[..., str]]:
        """Returns the tools as plain typed functions with docstrings."""

        def read_pages(pages: str, doc_id: str = "") -> str:
            """Reads pages of the document, e.g. pages="10-14,40".

            Cite answers as [doc_id p.N] using the markers in the reply.
            """
            return self.read(pages, doc_id)

        def search_document(query: str, k: int = 5, doc_id: str = "") -> str:
            """Finds the pages most relevant to a query, with snippets."""
            return self.search(query, k, doc_id)

        def document_outline(doc_id: str = "") -> str:
            """Lists the document's headings with their page numbers."""
            return self.outline(doc_id)

        return [read_pages, search_document, document_outline]
