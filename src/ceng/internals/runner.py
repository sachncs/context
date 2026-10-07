"""The single bridge from synchronous callers into the async core."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


def run_sync(coroutine: Coroutine[object, object, T]) -> T:
    """Runs `coroutine` to completion from synchronous code.

    Uses `asyncio.run` when no event loop is running in this thread. When
    called from inside a running loop (e.g. Jupyter), the coroutine runs on a
    fresh loop in a worker thread so the caller's loop is not re-entered.

    Args:
        coroutine: The awaitable to drive.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()
