"""Lossless text partitioners."""

from __future__ import annotations

import abc
import dataclasses

from ceng import errors
from ceng.internals import registry
from ceng.tokenizers import base as tokenizer_base


@dataclasses.dataclass(frozen=True, slots=True)
class Partition:
    """A contiguous slice of a source text.

    Attributes:
        text: The slice.
        index: Zero-based position in document order.
        start: Character offset of the slice in the source.
        tokens: Token count of `text`.
    """

    text: str
    index: int
    start: int
    tokens: int

    @property
    def end(self) -> int:
        """Returns the exclusive end offset in the source."""
        return self.start + len(self.text)


class Partitioner(abc.ABC):
    """Splits text into ordered, non-overlapping, lossless partitions.

    Invariant: concatenating `partition.text` over the result reproduces the
    input exactly.
    """

    registry: registry.Registry[type[Partitioner]] = registry.Registry(
        "partitioner"
    )

    @abc.abstractmethod
    def split(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[Partition]:
        """Partitions `text`.

        Raises:
            ValidationError: If `text` is empty.
        """

    @staticmethod
    def build(
        pieces: list[str], tokenizer: tokenizer_base.Tokenizer
    ) -> list[Partition]:
        """Wraps consecutive text pieces as `Partition`s with offsets."""
        partitions = []
        offset = 0
        for index, piece in enumerate(pieces):
            partitions.append(
                Partition(piece, index, offset, tokenizer.count(piece))
            )
            offset += len(piece)
        return partitions


SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def split_keeping(text: str, separator: str) -> list[str]:
    """Splits on `separator`, attaching it to the preceding unit."""
    pieces = text.split(separator)
    units = [piece + separator for piece in pieces[:-1]]
    if pieces[-1]:
        units.append(pieces[-1])
    return units


@Partitioner.registry.register("recursive")
class RecursivePartitioner(Partitioner):
    """Packs the coarsest natural units that fit, recursing on oversize ones.

    Tries paragraph, line, sentence, clause and word boundaries in turn and
    falls back to token-level cuts for unbroken text. Separators stay attached
    to their unit, so whitespace and line structure are preserved.

    Attributes:
        max_tokens: Hard ceiling for every partition.
        separators: Boundary strings, coarsest first.
    """

    def __init__(
        self, max_tokens: int = 512, separators: tuple[str, ...] = SEPARATORS
    ) -> None:
        """Creates the partitioner.

        Raises:
            ConfigError: If `max_tokens` is not positive.
        """
        if max_tokens < 1:
            raise errors.ConfigError("max_tokens must be >= 1")
        self.max_tokens = max_tokens
        self.separators = separators

    def split(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[Partition]:
        if not text:
            raise errors.ValidationError("cannot partition empty text")
        return self.build(self.pack(text, 0, tokenizer), tokenizer)

    def pack(
        self, text: str, level: int, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        """Returns pieces of `text`, each within `max_tokens`."""
        if tokenizer.count(text) <= self.max_tokens:
            return [text]
        if level >= len(self.separators):
            return self.hard_cut(text, tokenizer)
        units = split_keeping(text, self.separators[level])
        if len(units) <= 1:
            return self.pack(text, level + 1, tokenizer)
        pieces: list[str] = []
        current = ""
        for unit in units:
            if current and tokenizer.count(current + unit) > self.max_tokens:
                pieces.extend(self.pack(current, level + 1, tokenizer))
                current = unit
            else:
                current += unit
        if current:
            pieces.extend(self.pack(current, level + 1, tokenizer))
        return pieces

    def hard_cut(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        """Cuts unbroken text at token capacity, always making progress."""
        pieces = []
        rest = text
        while rest:
            head = tokenizer.truncate(rest, self.max_tokens) or rest[:1]
            pieces.append(head)
            rest = rest[len(head) :]
        return pieces


@Partitioner.registry.register("fixed")
class FixedWindowPartitioner(Partitioner):
    """Cuts text into consecutive windows of exactly `max_tokens` tokens."""

    def __init__(self, max_tokens: int = 512) -> None:
        """Creates the partitioner.

        Raises:
            ConfigError: If `max_tokens` is not positive.
        """
        if max_tokens < 1:
            raise errors.ConfigError("max_tokens must be >= 1")
        self.cutter = RecursivePartitioner(max_tokens, separators=())

    def split(
        self, text: str, tokenizer: tokenizer_base.Tokenizer
    ) -> list[Partition]:
        if not text:
            raise errors.ValidationError("cannot partition empty text")
        return self.build(self.cutter.hard_cut(text, tokenizer), tokenizer)
