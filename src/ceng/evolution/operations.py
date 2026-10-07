"""Curator operations: typed edits the Curator LLM proposes."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from typing import ClassVar

from ceng.evolution import playbook as playbook_lib
from ceng.internals import registry


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Result of applying an operation.

    Attributes:
        playbook: The updated playbook.
        added: Bullets added.
        removed: Bullets removed.
        label: Short description for step statistics.
    """

    playbook: playbook_lib.Playbook
    added: int = 0
    removed: int = 0
    label: str = ""


class CuratorOp(abc.ABC):
    """An edit to a playbook. Subclasses are frozen dataclasses."""

    registry: ClassVar[registry.Registry[type[CuratorOp]]] = registry.Registry(
        "curator operation"
    )
    kind: ClassVar[str] = ""

    @abc.abstractmethod
    def apply(self, playbook: playbook_lib.Playbook, now: str) -> Outcome:
        """Applies the edit; invalid targets are skipped, never raised."""

    @classmethod
    @abc.abstractmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CuratorOp | None:
        """Builds the op from model JSON, or None if it is unusable."""

    @staticmethod
    def parse(data: object) -> CuratorOp | None:
        """Dispatches a raw operation mapping to its subclass."""
        if not isinstance(data, Mapping):
            return None
        kind = str(data.get("type") or "").upper()
        if kind not in CuratorOp.registry:
            return None
        return CuratorOp.registry.get(kind).from_mapping(data)


def text_field(data: Mapping[str, object], *names: str) -> str:
    """Returns the first non-empty string among `names`, stripped."""
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@CuratorOp.registry.register("ADD")
@dataclasses.dataclass(frozen=True, slots=True)
class AddOp(CuratorOp):
    """Adds a bullet (de-duplicated against existing content)."""

    section: str
    content: str
    kind: ClassVar[str] = "ADD"

    def apply(self, playbook: playbook_lib.Playbook, now: str) -> Outcome:
        bullet = playbook_lib.Bullet.new(self.section, self.content, now)
        merged = playbook.merge([bullet])
        added = len(merged) - len(playbook)
        return Outcome(merged, added=added, label=f"ADD:{bullet.section}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> AddOp | None:
        content = text_field(data, "content")
        if not content:
            return None
        return cls(text_field(data, "section") or "others", content)


@CuratorOp.registry.register("UPDATE")
@dataclasses.dataclass(frozen=True, slots=True)
class UpdateOp(CuratorOp):
    """Rewrites the content of an existing bullet."""

    target: str
    content: str
    kind: ClassVar[str] = "UPDATE"

    def apply(self, playbook: playbook_lib.Playbook, now: str) -> Outcome:
        existing = playbook.get(self.target)
        if existing is None:
            return Outcome(playbook)
        updated = dataclasses.replace(
            existing, content=self.content, updated_at=now
        )
        return Outcome(playbook.replace(updated), label=f"UPDATE:{self.target}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> UpdateOp | None:
        target = text_field(data, "id", "bullet_id")
        content = text_field(data, "content")
        return cls(target, content) if target and content else None


@CuratorOp.registry.register("MERGE")
@dataclasses.dataclass(frozen=True, slots=True)
class MergeOp(CuratorOp):
    """Collapses two bullets; the higher-scoring one survives with the sum."""

    keep: str
    drop: str
    kind: ClassVar[str] = "MERGE"

    def apply(self, playbook: playbook_lib.Playbook, now: str) -> Outcome:
        keep, drop = playbook.get(self.keep), playbook.get(self.drop)
        if keep is None or drop is None:
            return Outcome(playbook)
        winner, loser = (
            (keep, drop) if keep.net_score >= drop.net_score else (drop, keep)
        )
        combined = dataclasses.replace(
            winner,
            helpful_count=winner.helpful_count + loser.helpful_count,
            harmful_count=winner.harmful_count + loser.harmful_count,
            updated_at=now,
        )
        result = playbook.replace(combined).without([loser.id])
        return Outcome(result, removed=1, label=f"MERGE:{keep.id}<-{drop.id}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> MergeOp | None:
        keep = text_field(data, "keep_id", "id")
        drop = text_field(data, "drop_id", "other_id")
        return cls(keep, drop) if keep and drop and keep != drop else None


@CuratorOp.registry.register("DELETE")
@dataclasses.dataclass(frozen=True, slots=True)
class DeleteOp(CuratorOp):
    """Removes a bullet."""

    target: str
    kind: ClassVar[str] = "DELETE"

    def apply(self, playbook: playbook_lib.Playbook, now: str) -> Outcome:
        if playbook.get(self.target) is None:
            return Outcome(playbook)
        return Outcome(
            playbook.without([self.target]),
            removed=1,
            label=f"DELETE:{self.target}",
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> DeleteOp | None:
        target = text_field(data, "id", "bullet_id")
        return cls(target) if target else None
