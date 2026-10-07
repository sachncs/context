"""Per-event-loop storage for objects that are bound to a loop."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class LoopLocal(Generic[T]):
    """Lazily creates one value per running event loop.

    Async clients, semaphores and locks must not be shared between event
    loops (the sync API runs each call in a fresh loop). Values are held
    weakly by loop, so a finished loop's value is released and a recycled
    object id can never hand a stale value to a new loop.
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        """Creates the store.

        Args:
            factory: Builds a value for a loop that has none yet.
        """
        self.factory = factory
        self.values: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = (
            weakref.WeakKeyDictionary()
        )

    def get(self) -> T:
        """Returns the running loop's value, creating it if needed."""
        loop = asyncio.get_running_loop()
        if loop not in self.values:
            self.values[loop] = self.factory()
        return self.values[loop]

    def peek(self) -> T | None:
        """Returns the running loop's value without creating one."""
        return self.values.get(asyncio.get_running_loop())

    def discard(self) -> T | None:
        """Removes and returns the running loop's value."""
        return self.values.pop(asyncio.get_running_loop(), None)
