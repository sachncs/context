"""Verifier interface: checks that a context is trustworthy or in-spec."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from ceng.internals import registry

if TYPE_CHECKING:
    from ceng import context as context_lib


@dataclasses.dataclass(frozen=True, slots=True)
class Verdict:
    """Outcome of a verification.

    Attributes:
        verifier: Name of the verifier that produced the verdict.
        passed: Whether the check succeeded.
        detail: One-line human explanation.
    """

    verifier: str
    passed: bool
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class Verifier(abc.ABC):
    """A check applied to a `Context`. Subclasses are frozen dataclasses."""

    registry: ClassVar[registry.Registry[type[Verifier]]] = registry.Registry(
        "verification method"
    )
    name: ClassVar[str] = ""

    @classmethod
    def register(cls, name: str) -> Callable[[type[Verifier]], type[Verifier]]:
        """Returns a class decorator registering a verifier under `name`."""

        def decorator(subclass: type[Verifier]) -> type[Verifier]:
            cls.registry.add(name, subclass)
            subclass.name = name
            return subclass

        return decorator

    @abc.abstractmethod
    async def verify(self, context: context_lib.Context) -> Verdict:
        """Runs the check against `context`."""
