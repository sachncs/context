"""Splitting a token budget across named context slots."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from foveate import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Slot:
    """A part of the prompt that competes for the budget.

    Attributes:
        name: Slot name ("instructions", "history", "evidence", ...).
        tokens: Exact size for a fixed slot; ignored when `weight` is set.
        weight: Relative share of what fixed slots leave over (flexible).
        minimum: Smallest allocation a flexible slot may receive.
    """

    name: str
    tokens: int = 0
    weight: float = 0.0
    minimum: int = 0


def allocate(budget: int, slots: Sequence[Slot]) -> dict[str, int]:
    """Allocates `budget` tokens across slots.

    Fixed slots (no weight) are served first. What remains is shared among
    flexible slots in proportion to their weights, never below each slot's
    `minimum`; if the minimums cannot all be met the budget error is raised
    rather than silently overrunning.

    Args:
        budget: Total tokens available.
        slots: The slots, in any order.

    Returns:
        Slot name to allocated tokens; the sum never exceeds `budget`.

    Raises:
        ConfigError: For duplicate names, or when fixed slots and minimums
            exceed the budget.
    """
    names = [s.name for s in slots]
    if len(set(names)) != len(names):
        raise errors.ConfigError("slot names must be unique")
    fixed = {s.name: s.tokens for s in slots if s.weight <= 0}
    flexible = [s for s in slots if s.weight > 0]
    remaining = budget - sum(fixed.values())
    if remaining < sum(s.minimum for s in flexible):
        raise errors.ConfigError(
            f"budget {budget} cannot fit the fixed slots plus minimums"
        )
    allocation = dict(fixed)
    pool = list(flexible)
    while pool:
        total_weight = sum(s.weight for s in pool)
        shares = {
            s.name: int(remaining * s.weight / total_weight) for s in pool
        }
        starved = [s for s in pool if shares[s.name] < s.minimum]
        if not starved:
            allocation.update(shares)
            break
        for slot in starved:
            allocation[slot.name] = slot.minimum
            remaining -= slot.minimum
            pool.remove(slot)
    return allocation
