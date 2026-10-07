"""Bounded, fail-fast concurrent execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[T]]], limit: int
) -> list[T]:
    """Runs awaitable factories with at most `limit` in flight.

    Results are returned in input order. On the first failure every other
    task is cancelled and awaited before the exception propagates, so no
    work leaks past the call.

    Args:
        factories: Zero-argument callables producing awaitables.
        limit: Maximum concurrent awaitables (>= 1).

    Returns:
        Results in the order of `factories`.
    """
    semaphore = asyncio.Semaphore(max(1, limit))

    async def guarded(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks = [asyncio.ensure_future(guarded(f)) for f in factories]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
