"""Foveation: spend the token budget where relevance is highest.

The eye resolves fine detail only at the fovea and coarse detail around it.
`allocate` does the same for a document set: the most relevant pages are kept
in full (fovea), their neighbours and the next-best pages are condensed
(parafovea), and every other page is reduced to one outline line
(periphery) so the model still knows what exists and can ask for it.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence

from foveate import errors
from foveate.compression import extractive, queryaware
from foveate.documents import document as document_lib
from foveate.tokenizers import base as tokenizer_base

PageKey = tuple[str, int]
OUTLINE_PREFIX_TOKENS = 4
ORDERS = ("reading", "edges")


class Tier(str, enum.Enum):
    """Fidelity a page is kept at."""

    FULL = "full"
    CONDENSED = "condensed"
    OUTLINE = "outline"
    DROPPED = "dropped"


@dataclasses.dataclass(frozen=True, slots=True)
class FoveationConfig:
    """Tunable shape of the allocation.

    Attributes:
        fovea_share: Fraction of the budget for FULL pages.
        parafovea_share: Fraction for CONDENSED pages; the rest of the budget
            holds the outline.
        neighbours: Pages on each side of a FULL page that are condensed.
        condensed_tokens: Target size of one condensed page.
        outline_tokens: Size cap of one outline line.
        max_full: Optional cap on the number of FULL pages.
        order: How shown pages are arranged in the prompt: `"reading"`
            (document order) or `"edges"` (best pages first and last, weakest
            in the middle, since models use the middle of a long prompt least).
        query_aware: Condense neighbour pages by keeping the sentences that
            match the question rather than the most frequent words.
    """

    fovea_share: float = 0.6
    parafovea_share: float = 0.25
    neighbours: int = 1
    condensed_tokens: int = 150
    outline_tokens: int = 24
    max_full: int | None = None
    order: str = "reading"
    query_aware: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.fovea_share < 1 or not 0 <= self.parafovea_share < 1:
            raise errors.ConfigError("shares must be inside (0, 1)")
        if self.fovea_share + self.parafovea_share >= 1:
            raise errors.ConfigError(
                "fovea_share + parafovea_share must leave room for the outline"
            )
        if self.neighbours < 0 or self.condensed_tokens < 8:
            raise errors.ConfigError("invalid neighbours or condensed_tokens")
        if self.outline_tokens < 4:
            raise errors.ConfigError("outline_tokens must be >= 4")
        if self.max_full is not None and self.max_full < 1:
            raise errors.ConfigError("max_full must be >= 1")
        if self.order not in ORDERS:
            raise errors.ConfigError(f"order must be one of {ORDERS}")


@dataclasses.dataclass(frozen=True, slots=True)
class PagePlan:
    """The fate of one page.

    Attributes:
        doc_id: Document id.
        page: Page number.
        tier: Fidelity chosen.
        score: Relevance score that earned the tier (0 for neighbours).
        tokens_before: Original page tokens.
        tokens_after: Tokens the page costs at this tier.
        text: Text to show ("" for DROPPED).
    """

    doc_id: str
    page: int
    tier: Tier
    score: float
    tokens_before: int
    tokens_after: int
    text: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Foveation:
    """A complete allocation.

    Attributes:
        pages: One plan per page of every document, in reading order.
        budget: Token budget it was built for.
    """

    pages: tuple[PagePlan, ...]
    budget: int

    @property
    def tokens(self) -> int:
        """Returns the total tokens the allocation will occupy."""
        return sum(p.tokens_after for p in self.pages)

    def tier(self, tier: Tier) -> list[PagePlan]:
        """Returns the pages assigned to `tier`."""
        return [p for p in self.pages if p.tier is tier]

    def shown(self) -> list[PagePlan]:
        """Returns FULL and CONDENSED pages (those with body text)."""
        return [p for p in self.pages if p.tier in (Tier.FULL, Tier.CONDENSED)]


def arrange(pages: Sequence[PagePlan], order: str) -> list[PagePlan]:
    """Orders shown pages for the prompt.

    `"reading"` keeps document order. `"edges"` ranks FULL pages by score and
    deals them alternately to the front and the back, so the best evidence sits
    at both ends; CONDENSED pages (neighbours) fill the middle in reading
    order.
    """
    if order == "reading":
        return list(pages)
    full = sorted(
        (p for p in pages if p.tier is Tier.FULL), key=lambda p: -p.score
    )
    middle = [p for p in pages if p.tier is not Tier.FULL]
    front, back = full[0::2], full[1::2]
    return [*front, *middle, *reversed(back)]


def outline_line(
    document: document_lib.Document,
    number: int,
    cap: int,
    tokenizer: tokenizer_base.Tokenizer,
) -> str:
    """Builds `"doc p.N: heading or first words"` capped to `cap` tokens."""
    page = document.page(number)
    head = page.headings[0] if page.headings else " ".join(page.text.split())
    label = f"{document.id} p.{number}: "
    return label + tokenizer.truncate(head, max(1, cap - OUTLINE_PREFIX_TOKENS))


def allocate(
    ranked: Sequence[tuple[PageKey, float]],
    documents: Sequence[document_lib.Document],
    budget: int,
    tokenizer: tokenizer_base.Tokenizer,
    config: FoveationConfig | None = None,
    query: str = "",
) -> Foveation:
    """Assigns every page a tier so the result fits `budget` tokens.

    Args:
        ranked: `((doc_id, page), score)` pairs, most relevant first.
        documents: The documents the pages come from.
        budget: Token ceiling for the whole allocation.
        tokenizer: Token counter.
        config: Shape of the allocation.
        query: The question; used to condense neighbours when
            `config.query_aware` is set.

    Returns:
        A `Foveation` whose `tokens` never exceed `budget`.

    Raises:
        ConfigError: For a non-positive budget.
    """
    if budget < 1:
        raise errors.ConfigError("budget must be >= 1")
    cfg = config or FoveationConfig()
    docs: Mapping[str, document_lib.Document] = {d.id: d for d in documents}
    plans: dict[PageKey, PagePlan] = {}
    condenser: extractive.Extractive = (
        queryaware.QueryExtractive(query=query)
        if cfg.query_aware and query
        else extractive.Extractive()
    )

    def cost(key: PageKey) -> int:
        return docs[key[0]].page(key[1]).tokens

    full_budget = int(budget * cfg.fovea_share)
    spent = 0
    for key, score in ranked:
        if key in plans or key[0] not in docs:
            continue
        if cfg.max_full is not None and len(plans) >= cfg.max_full:
            break
        page = docs[key[0]].page(key[1])
        need = page.tokens
        if need > full_budget - spent:
            if spent or need <= full_budget:
                continue  # skip pages that do not fit; smaller ones may
            text = tokenizer.truncate(page.text, full_budget)
            need = tokenizer.count(text)
        else:
            text = page.text
        plans[key] = PagePlan(
            key[0], key[1], Tier.FULL, score, page.tokens, need, text
        )
        spent += need

    full_keys = list(plans)
    condensed_budget = int(budget * cfg.parafovea_share)
    candidates: list[tuple[PageKey, float]] = []
    for key in full_keys:
        document = docs[key[0]]
        for offset in range(1, cfg.neighbours + 1):
            for number in (key[1] - offset, key[1] + offset):
                if number in document.numbers:
                    candidates.append(((key[0], number), 0.0))
    candidates.extend(ranked)
    used = 0
    for key, score in candidates:
        if key in plans or key[0] not in docs:
            continue
        page = docs[key[0]].page(key[1])
        if not page.text.strip():
            continue
        target = min(cfg.condensed_tokens, page.tokens)
        if used + target > condensed_budget:
            continue
        text = (
            page.text
            if page.tokens <= target
            else condenser.extract(page.text, target, tokenizer)
        )
        size = tokenizer.count(text)
        if used + size > condensed_budget:
            continue
        plans[key] = PagePlan(
            key[0], key[1], Tier.CONDENSED, score, page.tokens, size, text
        )
        used += size

    outline_budget = budget - spent - used
    for document in documents:
        for page in document.pages:
            key = (document.id, page.number)
            if key in plans:
                continue
            line = outline_line(
                document, page.number, cfg.outline_tokens, tokenizer
            )
            size = tokenizer.count(line) + 1
            if size <= outline_budget and page.text.strip():
                plans[key] = PagePlan(
                    key[0], key[1], Tier.OUTLINE, 0.0, page.tokens, size, line
                )
                outline_budget -= size
            else:
                plans[key] = PagePlan(
                    key[0], key[1], Tier.DROPPED, 0.0, page.tokens, 0, ""
                )
    order = [(d.id, p.number) for d in documents for p in d.pages]
    return Foveation(tuple(plans[key] for key in order), budget)
