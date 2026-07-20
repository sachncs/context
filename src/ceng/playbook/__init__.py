"""ceng.playbook — ACE-style evolving bullet playbook package.

Zhang et al. (arXiv:2510.04618) ACE: treat context as an itemized
collection of bullets that accumulate, refine, and organize strategies
over time. Each bullet is ``[id] helpful=N harmful=N :: content``.

Surface:

- :class:`Bullet` — id, content, helpful_count, harmful_count
- :class:`Playbook` — sectioned bullet collection with deterministic
  merge + content-hash dedup
- :func:`parse_playbook` / :func:`render_playbook` — round-trip
  with the on-disk line format
- :class:`~ceng.playbook.evolver.Evolver` — the Generator → Reflector
  → Curator loop (in ``ceng.playbook.evolver``)

Verbatim Generator / Reflector / Curator prompts from the
upstream ACE GitHub repo (Stanford / SambaNova / UC Berkeley,
MIT) live in :mod:`ceng.playbook.prompts`.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_SECTIONS: tuple[str, ...] = (
    "strategies_and_insights",
    "formulas_and_calculations",
    "code_snippets_and_templates",
    "common_mistakes_to_avoid",
    "problem_solving_heuristics",
    "context_clues_and_indicators",
    "others",
)


@dataclass(frozen=True)
class Bullet:
    """A single ACE playbook bullet.

    Attributes:
        id: unique slug, e.g. ``str-00042``. The section prefix and
            the numeric suffix determine ordering.
        section: Section name (lowercase, spaces -> underscores).
        content: The strategy / formula / mistake etc.
        helpful_count: How many Reflector runs marked this bullet
            helpful.
        harmful_count: How many Reflector runs marked this bullet
            harmful.
        created_at: ISO 8601 timestamp.
        updated_at: ISO 8601 timestamp.
    """

    id: str
    section: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def net_score(self) -> int:
        """Helpful minus harmful. Use this to rank bullets."""
        return self.helpful_count - self.harmful_count


# Match: [id] helpful=N harmful=N :: content
_BULLET_RE = re.compile(
    r"^\[([A-Za-z0-9_\-]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)$"
)
_SECTION_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$")


@dataclass
class Playbook:
    """In-memory playbook of ACE bullets, sectioned, ordered.

    Mutations are append-only via ``add_bullet`` and ``merge`` (which
    deduplicates by content hash, preferring the higher net score).
    A :class:`Playbook` is reconstructable to text via
    ``render_for_context`` or to a list of bullets via ``get_active``.
    """

    bullets: dict[str, Bullet] = field(default_factory=OrderedDict)
    sections_in_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for section in DEFAULT_SECTIONS:
            if section not in self.sections_in_order:
                self.sections_in_order.append(section)

    @classmethod
    def from_text(cls, text: str) -> "Playbook":
        """Parse an ACE-flavoured playbook (one ``## Section\\n[id] ::\\n`` per bullet).

        Unknown sections are appended after the default ones so the
        produced playbook stays readable in a known order.
        """
        p = cls()
        current_section = "others"
        if not text.strip():
            return p
        for raw in text.splitlines():
            line = raw.rstrip()
            sec = _SECTION_HEADER_RE.match(line)
            if sec:
                name = canon_section(sec.group(1))
                if name not in p.sections_in_order:
                    p.sections_in_order.append(name)
                current_section = name
                continue
            b = _BULLET_RE.match(line)
            if b:
                bullet = Bullet(
                    id=b.group(1),
                    section=current_section,
                    content=b.group(4).strip(),
                    helpful_count=int(b.group(2)),
                    harmful_count=int(b.group(3)),
                )
                p.bullets[bullet.id] = bullet
        return p

    def add_bullet(self, bullet: Bullet) -> None:
        if bullet.section not in self.sections_in_order:
            self.sections_in_order.append(canon_section(bullet.section))
        self.bullets[bullet.id] = bullet

    def merge(
        self, incoming: Iterable[Bullet], *, dedup_threshold: float = 0.90
    ) -> int:
        """Merge ``incoming`` bullets into the playbook.

        Two-pass dedup:
        1. Bullet-id collision: keep the higher net_score.
        2. Content-hash dedup: bullets with the same
           normalised content keep the higher net score.

        Returns the number of NEW bullets added (existing bullets
        that were merged are not counted).
        """
        added = 0
        for bullet in incoming:
            existing = self.bullets.get(bullet.id)
            if existing is not None:
                if bullet.net_score > existing.net_score:
                    self.bullets[bullet.id] = bullet
                continue
            canon = content_canonical(bullet.content)
            dup_id = self._find_duplicate_by_content(canon)
            if dup_id is not None:
                existing = self.bullets[dup_id]
                if bullet.net_score > existing.net_score:
                    self.bullets[dup_id] = bullet
                continue
            self.bullets[bullet.id] = bullet
            added += 1
        return added

    def _find_duplicate_by_content(self, canonical: str) -> str | None:
        for bid, b in self.bullets.items():
            if content_canonical(b.content) == canonical:
                return bid
        return None

    def get_active(self, top_k: int | None = None) -> list[Bullet]:
        """Return bullets ordered by ``net_score`` desc, then ``id`` asc."""
        bullets = sorted(
            self.bullets.values(),
            key=lambda b: (-b.net_score, b.id),
        )
        if top_k is not None:
            return bullets[:top_k]
        return bullets

    def trim_to_token_budget(
        self, tokenizer=None, budget: int = 80_000
    ) -> int:
        """Drop lowest-net-score bullets until the rendered playbook
        fits under ``budget`` tokens. Returns the number dropped.

        Always keeps at least the empty-playbook skeleton (one per
        section) so the rendered text never goes invalid.
        """
        from ceng.tokens import count_tokens

        current_dropped = 0
        while True:
            text = self.render_for_context()
            if count_tokens(text, tokenizer) <= budget:
                return current_dropped
            active = self.get_active()
            if not active:
                return current_dropped
            victim = active[-1]
            del self.bullets[victim.id]
            current_dropped += 1

    def render_for_context(self) -> str:
        """Render the playbook as ``## Section\\n[id] :: text\\n…`` text."""
        by_section: dict[str, list[Bullet]] = {s: [] for s in self.sections_in_order}
        for bullet in self.bullets.values():
            by_section.setdefault(bullet.section, []).append(bullet)
        lines: list[str] = []
        for section in self.sections_in_order:
            bullets = by_section.get(section) or []
            title = section.replace("_", " ").title()
            lines.append(f"## {title}")
            bullets.sort(key=lambda b: b.id)
            for b in bullets:
                lines.append(format_bullet(b))
            lines.append("")
        return "\n".join(lines)


def empty_playbook() -> Playbook:
    """Return a Playbook seeded with the seven ACE default sections."""
    return Playbook()


def parse_playbook(text: str) -> Playbook:
    return Playbook.from_text(text)


def render_playbook(playbook: Playbook) -> str:
    return playbook.render_for_context()


def format_bullet(bullet: Bullet) -> str:
    return (
        f"[{bullet.id}] helpful={bullet.helpful_count} "
        f"harmful={bullet.harmful_count} :: {bullet.content}"
    )


def canon_section(name: str) -> str:
    """Normalise a section header to its canonical slug.

    Maps any flavour of header casing / separators onto the same
    key. The seven canonical names are::

        strategies_and_insights
        formulas_and_calculations
        code_snippets_and_templates
        common_mistakes_to_avoid
        problem_solving_heuristics
        context_clues_and_indicators
        others
    """
    s = name.strip().lower()
    return s.replace("&", "and").replace(" ", "_").replace("-", "_")


def content_canonical(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


# Re-export Evolver from the sibling module so callers can do
# ``from ceng.playbook import Evolver``.
from ceng.playbook.evolver import Evolver, EvolverConfig, EvolverStepStats  # noqa: E402, F401

__all__ = [
    "Bullet",
    "Playbook",
    "empty_playbook",
    "parse_playbook",
    "render_playbook",
    "format_bullet",
    "canon_section",
    "content_canonical",
    "DEFAULT_SECTIONS",
    "Evolver",
    "EvolverConfig",
    "EvolverStepStats",
]
