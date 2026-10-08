"""Pages, headings and page-range parsing."""

from __future__ import annotations

import dataclasses
import re

from foveate import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Page:
    """One page of a document.

    Attributes:
        number: 1-based page number as the reader sees it. Subsets of a
            document keep the original numbers, so citations stay valid.
        text: Extracted text (may be empty for scanned pages).
        tokens: Token count of `text`.
        headings: Headings detected on the page, in order.
    """

    number: int
    text: str
    tokens: int
    headings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.number < 1:
            raise errors.ValidationError("page numbers start at 1")


@dataclasses.dataclass(frozen=True, slots=True)
class Heading:
    """An outline entry.

    Attributes:
        title: Heading text.
        page: Page the heading appears on.
    """

    title: str
    page: int


PAGE_PART = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_pages(
    spec: str | int | range, available: list[int]
) -> tuple[int, ...]:
    """Expands a page specification into sorted, unique page numbers.

    Args:
        spec: `"10-14,40"`, a single integer, or a `range`.
        available: Page numbers that exist; requests outside it are errors.

    Returns:
        Sorted unique page numbers.

    Raises:
        ValidationError: For malformed specs, reversed ranges, or pages the
            document does not have.
    """
    if isinstance(spec, int):
        wanted = [spec]
    elif isinstance(spec, range):
        wanted = list(spec)
    else:
        wanted = []
        for part in spec.split(","):
            if not part.strip():
                continue
            match = PAGE_PART.match(part)
            if match is None:
                raise errors.ValidationError(f"invalid page spec: {part!r}")
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise errors.ValidationError(f"reversed page range: {part!r}")
            wanted.extend(range(start, end + 1))
    if not wanted:
        raise errors.ValidationError("empty page specification")
    known = set(available)
    missing = sorted(set(wanted) - known)
    if missing:
        raise errors.ValidationError(
            f"document has no page(s) {missing}; "
            f"it has {min(known)}-{max(known)}"
        )
    return tuple(sorted(set(wanted)))


NUMBERED = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+\.|Item\s+\d+[A-C]?\.?)\s+\S"
)
MARKDOWN = re.compile(r"^#{1,6}\s+\S")
MAX_HEADING_CHARS = 90


def detect_headings(text: str) -> tuple[str, ...]:
    """Finds likely headings with cheap, language-agnostic heuristics.

    Recognises Markdown `#` headings, numbered sections ("2.1 Scope",
    "Item 7."), and short ALL-CAPS lines. Precision matters more than
    recall: a wrong heading only affects the optional outline.
    """
    found = []
    for raw in text.splitlines():
        line = raw.strip()
        if not 3 <= len(line) <= MAX_HEADING_CHARS or line.endswith((",", ";")):
            continue
        letters = [c for c in line if c.isalpha()]
        shouting = len(letters) >= 4 and all(c.isupper() for c in letters)
        if MARKDOWN.match(line):
            found.append(line.lstrip("#").strip())
        elif (NUMBERED.match(line) and len(line.split()) <= 12) or (
            shouting and not line.isdigit()
        ):
            found.append(line)
    return tuple(found)
