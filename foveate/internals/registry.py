"""Generic name -> class registry used by every extension point."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from foveate import errors

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps unique string names to registered classes or factories.

    A registry is an instance owned by the extension point (for example
    `Compressor.registry`), never a module-level global.
    """

    def __init__(self, kind: str) -> None:
        """Creates an empty registry.

        Args:
            kind: Noun used in error messages (e.g. "compression method").
        """
        self.kind = kind
        self.entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Returns a decorator registering the decorated object as `name`.

        Raises:
            ConfigError: At decoration time if `name` is already taken.
        """

        def decorator(entry: T) -> T:
            self.add(name, entry)
            return entry

        return decorator

    def add(self, name: str, entry: T) -> None:
        """Registers `entry` under `name`.

        Raises:
            ConfigError: If `name` is empty or already registered.
        """
        if not name:
            raise errors.ConfigError(f"{self.kind} name must be non-empty")
        if name in self.entries:
            raise errors.ConfigError(f"{self.kind} {name!r} already registered")
        self.entries[name] = entry

    def get(self, name: str) -> T:
        """Returns the entry registered as `name`.

        Raises:
            ConfigError: If no such entry exists.
        """
        try:
            return self.entries[name]
        except KeyError:
            known = ", ".join(sorted(self.entries)) or "(none)"
            raise errors.ConfigError(
                f"unknown {self.kind} {name!r}; known: {known}"
            ) from None

    def names(self) -> list[str]:
        """Returns registered names, sorted."""
        return sorted(self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
