"""Portable cross-process lock based on exclusive file creation."""

from __future__ import annotations

import contextlib
import os
import pathlib
import time
from collections.abc import Iterator

from ceng import errors

POLL_SECONDS = 0.01


@contextlib.contextmanager
def file_lock(
    path: pathlib.Path, timeout: float = 10.0, stale_after: float = 60.0
) -> Iterator[None]:
    """Holds an exclusive lock identified by the lock file `path`.

    A lock file older than `stale_after` seconds is treated as abandoned by
    a crashed holder and is removed.

    Args:
        path: Lock file location (parent directory must exist).
        timeout: Seconds to wait before giving up.
        stale_after: Age at which an existing lock is considered abandoned.

    Raises:
        ConfigError: If the lock cannot be acquired within `timeout`.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > stale_after:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise errors.ConfigError(
                    f"could not acquire lock {path}"
                ) from None
            time.sleep(POLL_SECONDS)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
