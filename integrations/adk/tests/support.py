"""Shared helpers for the adapter tests (no model, no key needed)."""

from foveate import Runtime
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.tokenizers import HeuristicTokenizer

LONG = "word " * 120
TOK = HeuristicTokenizer()


def make_runtime():
    """Returns a runtime that raises if a model is ever called."""
    return Runtime.without_llm(), None


def make_doc(pages=30):
    """Builds a document whose page 7 holds a known fact."""
    items = []
    for n in range(1, pages + 1):
        fact = (
            " Capital expenditures were 1,577 million dollars."
            if n == 7
            else ""
        )
        body = (
            f"Routine discussion of operations on page {n}.{fact}"
            + " Filler." * 60
        )
        items.append(page_lib.Page(n, body, TOK.count(body), (f"SECTION {n}",)))
    return Document("acme_2018", tuple(items))
