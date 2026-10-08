"""Where haystack text comes from, and the public needle sets.

* `ProseSource`: seeded synthetic prose (no download).
* `BookSource`: real text, for example a NoLiMa book.
* `load_nolima`: the NoLiMa needle templates and a book, downloaded on first
  use into the benchmark cache (Adobe Research License: noncommercial
  research use; the files are not redistributed with Foveate).
"""

from __future__ import annotations

import abc
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from foveate import errors
from foveate.bench.longdoc import dataset
from foveate.bench.needle import haystack
from foveate.tokenizers import base as tokenizer_base

NOLIMA = "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/"
NEEDLE_SET = "needlesets/needle_set.json"
BOOK = "haystack/rand_shuffle/rand_book_1.txt"


class Source(abc.ABC):
    """Produces haystack pages of a requested size."""

    def __init__(self, tokenizer: tokenizer_base.Tokenizer) -> None:
        """Creates the source.

        Args:
            tokenizer: Counts tokens for page sizes.
        """
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def pages(self, tokens: int, seed: int) -> list[str]:
        """Returns pages totalling at least `tokens` tokens."""


class ProseSource(Source):
    """Seeded synthetic prose."""

    def pages(self, tokens: int, seed: int) -> list[str]:
        return haystack.prose_pages(tokens, self.tokenizer, seed)


class BookSource(Source):
    """Pages cut from real text."""

    def __init__(self, tokenizer: tokenizer_base.Tokenizer, text: str) -> None:
        """Creates the source from the book's full text."""
        super().__init__(tokenizer)
        self.text = text

    def pages(self, tokens: int, seed: int) -> list[str]:
        return haystack.book_pages(self.text, tokens, self.tokenizer, seed)


def cached(name: str, url: str, home: pathlib.Path | None = None) -> bytes:
    """Returns `url`'s bytes, downloading into the cache on first use."""
    path = (home or dataset.cache_home()) / "needle" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(dataset.fetch(url))
    return path.read_bytes()


def load_nolima(
    tokenizer: tokenizer_base.Tokenizer, home: pathlib.Path | None = None
) -> tuple[Sequence[Mapping[str, Any]], BookSource]:
    """Returns NoLiMa's needle templates and a book as a haystack source.

    Raises:
        ValidationError: If a download fails or the file is malformed.
    """
    needles = json.loads(
        cached("nolima_needles.json", NOLIMA + NEEDLE_SET, home)
    )
    if not isinstance(needles, list):
        raise errors.ValidationError("NoLiMa needle set is not a list")
    book = cached("nolima_book1.txt", NOLIMA + BOOK, home).decode("utf-8")
    return needles, BookSource(tokenizer, book)
