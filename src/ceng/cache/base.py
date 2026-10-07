"""Cache interface and in-memory / null implementations."""

from __future__ import annotations

import abc
import threading
from types import TracebackType

from ceng.internals import registry


class Cache(abc.ABC):
    """String key/value store with optional expiry. Safe to share."""

    registry: registry.Registry[type[Cache]] = registry.Registry("cache")

    @abc.abstractmethod
    def get(self, key: str) -> str | None:
        """Returns the value for `key`, or None on miss/expiry/corruption."""

    @abc.abstractmethod
    def set(self, key: str, value: str) -> None:
        """Stores `value` under `key`, replacing any previous value."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Removes `key` if present."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Removes every entry."""

    def close(self) -> None:
        """Releases resources. Default: nothing to release."""

    def __enter__(self) -> Cache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@Cache.registry.register("memory")
class MemoryCache(Cache):
    """Process-local dictionary cache."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        with self.lock:
            self.data[key] = value

    def delete(self, key: str) -> None:
        with self.lock:
            self.data.pop(key, None)

    def clear(self) -> None:
        with self.lock:
            self.data.clear()


@Cache.registry.register("null")
class NullCache(Cache):
    """Caches nothing. Use to disable caching explicitly."""

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def clear(self) -> None:
        return None
