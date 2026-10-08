"""ACE playbook: an itemised, sectioned collection of strategy bullets.

Zhang et al. (arXiv:2510.04618) treat context as bullets that accumulate
and refine over time. Text form: `[id] helpful=N harmful=N :: content`
under `## Section` headers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from foveate import errors
from foveate import messages as messages_lib
from foveate.tokenizers import base as tokenizer_base

DEFAULT_SECTIONS = (
    "strategies_and_insights",
    "formulas_and_calculations",
    "code_snippets_and_templates",
    "common_mistakes_to_avoid",
    "problem_solving_heuristics",
    "context_clues_and_indicators",
    "others",
)
ID_NAMESPACE = uuid.UUID("6f3c2d6e-5a0b-4c53-9a43-0c7f1b2f9a11")
BULLET_LINE = re.compile(
    r"^\[([A-Za-z0-9_\-]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)$"
)
SECTION_LINE = re.compile(r"^##\s+(.+?)\s*$")


def canonical_section(name: str) -> str:
    """Maps any header spelling onto its canonical slug."""
    slug = name.strip().lower().replace("&", "and")
    return re.sub(r"[\s\-]+", "_", slug).strip("_") or "others"


def content_key(content: str) -> str:
    """Returns the digest used to detect duplicate content."""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


def bullet_id(section: str, content: str) -> str:
    """Derives a deterministic, collision-free id from content.

    The id is a UUIDv5 of the normalised content, prefixed by the section's
    initials, so equal content always yields the same id (reproducible
    cache keys) and different content effectively never collides.
    """
    initials = "".join(word[0] for word in section.split("_") if word)[:4]
    digest = uuid.uuid5(ID_NAMESPACE, content_key(content)).hex[:12]
    return f"{initials or 'b'}-{digest}"


@dataclasses.dataclass(frozen=True, slots=True)
class Bullet:
    """One playbook entry.

    Attributes:
        id: Unique identifier.
        section: Canonical section slug.
        content: The strategy, formula or mistake.
        helpful_count: Times the bullet was judged helpful.
        harmful_count: Times the bullet was judged harmful.
        created_at: ISO 8601 creation time (may be empty).
        updated_at: ISO 8601 last-change time (may be empty).
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
        """Returns helpful minus harmful."""
        return self.helpful_count - self.harmful_count

    @classmethod
    def new(cls, section: str, content: str, now: str = "") -> Bullet:
        """Creates a bullet with a content-derived id."""
        slug = canonical_section(section)
        text = content.strip()
        return cls(bullet_id(slug, text), slug, text, 0, 0, now, now)

    def render(self) -> str:
        """Returns the one-line text form."""
        return (
            f"[{self.id}] helpful={self.helpful_count} "
            f"harmful={self.harmful_count} :: {self.content}"
        )

    def to_mapping(self) -> dict[str, Any]:
        """Returns a JSON-serialisable mapping."""
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Bullet:
        """Rebuilds a bullet from `to_mapping` output.

        Raises:
            ValidationError: If a field is missing or mistyped.
        """
        try:
            return cls(
                **{f.name: data[f.name] for f in dataclasses.fields(cls)}
            )
        except (KeyError, TypeError) as exc:
            raise errors.ValidationError(f"invalid bullet: {exc}") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class Playbook:
    """An immutable set of bullets; every change returns a new playbook.

    Attributes:
        bullets: Bullets in insertion order.
        sections: Section slugs in display order (defaults always present).
    """

    bullets: tuple[Bullet, ...] = ()
    sections: tuple[str, ...] = DEFAULT_SECTIONS

    def __post_init__(self) -> None:
        sections = list(self.sections)
        for default in DEFAULT_SECTIONS:
            if default not in sections:
                sections.append(default)
        for bullet in self.bullets:
            if bullet.section not in sections:
                sections.append(bullet.section)
        object.__setattr__(self, "sections", tuple(sections))
        ids = [b.id for b in self.bullets]
        if len(set(ids)) != len(ids):
            raise errors.ValidationError("playbook has duplicate bullet ids")

    def __len__(self) -> int:
        return len(self.bullets)

    def get(self, bullet_id_: str) -> Bullet | None:
        """Returns the bullet with the given id, or None."""
        return next((b for b in self.bullets if b.id == bullet_id_), None)

    def active(self, top_k: int | None = None) -> list[Bullet]:
        """Returns bullets best first (net score desc, then id)."""
        ranked = sorted(self.bullets, key=lambda b: (-b.net_score, b.id))
        return ranked if top_k is None else ranked[:top_k]

    def replace(self, bullet: Bullet) -> Playbook:
        """Returns a playbook with `bullet` replacing the same id."""
        return dataclasses.replace(
            self,
            bullets=tuple(
                bullet if b.id == bullet.id else b for b in self.bullets
            ),
        )

    def without(self, ids: Iterable[str]) -> Playbook:
        """Returns a playbook lacking the bullets with the given ids."""
        doomed = set(ids)
        return dataclasses.replace(
            self, bullets=tuple(b for b in self.bullets if b.id not in doomed)
        )

    def merge(self, incoming: Iterable[Bullet]) -> Playbook:
        """Adds bullets, de-duplicating by id and by normalised content.

        When an incoming bullet duplicates an existing one (same id or same
        content) the higher net score wins; otherwise it is appended.
        """
        current = list(self.bullets)
        by_id = {b.id: i for i, b in enumerate(current)}
        by_content = {content_key(b.content): i for i, b in enumerate(current)}
        for bullet in incoming:
            slot = by_id.get(bullet.id)
            if slot is None:
                slot = by_content.get(content_key(bullet.content))
            if slot is not None:
                if bullet.net_score > current[slot].net_score:
                    current[slot] = dataclasses.replace(
                        bullet, id=current[slot].id
                    )
                continue
            by_id[bullet.id] = len(current)
            by_content[content_key(bullet.content)] = len(current)
            current.append(bullet)
        return dataclasses.replace(self, bullets=tuple(current))

    def trim(
        self, tokenizer: tokenizer_base.Tokenizer, budget: int
    ) -> Playbook:
        """Drops the lowest-scoring bullets until the rendering fits.

        Costs are computed once per bullet (linear time) instead of
        re-rendering after each drop; a final render check removes any
        residual overshoot.

        Args:
            tokenizer: Token counter.
            budget: Maximum tokens of `render()`.
        """
        kept = list(self.active())
        skeleton = tokenizer.count(
            dataclasses.replace(self, bullets=()).render()
        )
        costs = {b.id: tokenizer.count(b.render()) + 1 for b in kept}
        total = skeleton + sum(costs.values())
        while kept and total > budget:
            victim = kept.pop()
            total -= costs[victim.id]
        result = self.without(
            b.id for b in self.bullets if b.id not in {k.id for k in kept}
        )
        while result.bullets and tokenizer.count(result.render()) > budget:
            result = result.without([result.active()[-1].id])
        return result

    def render(self) -> str:
        """Renders `## Section` blocks with one bullet per line."""
        lines: list[str] = []
        for section in self.sections:
            lines.append(f"## {section.replace('_', ' ').title()}")
            lines.extend(
                b.render()
                for b in sorted(self.bullets, key=lambda b: b.id)
                if b.section == section
            )
            lines.append("")
        return "\n".join(lines)

    def stats(self) -> str:
        """Returns a one-line per-section bullet count summary."""
        counts = {s: 0 for s in self.sections}
        for bullet in self.bullets:
            counts[bullet.section] += 1
        parts = [f"total={len(self.bullets)}"]
        parts.extend(f"{s}={c}" for s, c in counts.items())
        return ", ".join(parts)

    def as_message(
        self, role: messages_lib.Role = messages_lib.Role.SYSTEM
    ) -> messages_lib.Message:
        """Wraps the rendered playbook as a message for a `Context`."""
        return messages_lib.Message(
            role, "Playbook of strategies:\n" + self.render()
        )

    @classmethod
    def parse(cls, text: str) -> Playbook:
        """Parses the text form; unknown sections are appended."""
        bullets: list[Bullet] = []
        sections: list[str] = list(DEFAULT_SECTIONS)
        current = "others"
        for raw in text.splitlines():
            line = raw.rstrip()
            header = SECTION_LINE.match(line)
            if header:
                current = canonical_section(header.group(1))
                if current not in sections:
                    sections.append(current)
                continue
            match = BULLET_LINE.match(line)
            if match:
                bullets.append(
                    Bullet(
                        match.group(1),
                        current,
                        match.group(4).strip(),
                        int(match.group(2)),
                        int(match.group(3)),
                    )
                )
        return cls(tuple(bullets), tuple(sections))

    def to_json(self) -> str:
        """Serialises to JSON (round-trips timestamps, unlike text form)."""
        return json.dumps(
            {
                "sections": list(self.sections),
                "bullets": [b.to_mapping() for b in self.bullets],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> Playbook:
        """Rebuilds a playbook from `to_json` output.

        Raises:
            ValidationError: If the JSON is malformed.
        """
        try:
            data = json.loads(text)
            return cls(
                tuple(Bullet.from_mapping(b) for b in data["bullets"]),
                tuple(data["sections"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise errors.ValidationError(
                f"invalid playbook JSON: {exc}"
            ) from exc
