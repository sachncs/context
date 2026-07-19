"""sqlite-backed cache for LLM call results.

The cache stores serialised values under deterministic keys. Two namespaces
keep concerns separate:

* ``"summarize"`` — leaf-context summaries produced by :mod:`ceng.compress`.
* ``"ppa_check"`` — leaf answers produced by :mod:`ceng.check`.

The cache uses only the Python standard library so it adds no install
weight. Key derivation uses SHA-256 over a canonical JSON dump, so two
callers producing the same arguments hit the same row without having to
serialise the value first.

Typical use::

    cache = Cache(cache_dir=".ceng_cache")
    cache.set("summarize", key="abc123", value={"summary": "..."})
    hit = cache.get("summarize", key="abc123")  # {"summary": "..."}
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1"
NAMESPACE_SUMMARIZE = "summarize"
NAMESPACE_PPA_CHECK = "ppa_check"

VALID_NAMESPACES = frozenset({NAMESPACE_SUMMARIZE, NAMESPACE_PPA_CHECK})


def make_key(parts: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for a dict of key parts.

    Args:
        parts: Mapping whose values are JSON-serialisable. Order does not
            affect the digest because the dict is serialised with sorted
            keys.

    Returns:
        64-character hex digest usable as a cache key.

    Raises:
        TypeError: If any value is not JSON-serialisable.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Cache:
    """On-disk key/value cache backed by sqlite.

    Thread-safe. One process owns the file. For multi-process sharing,
    point multiple :class:`Cache` instances at the same ``cache_dir``.

    Attributes:
        cache_dir: Directory the sqlite file lives in. Created on demand.
        db_name: Filename of the sqlite database inside ``cache_dir``.
    """

    cache_dir: str = ".ceng_cache"
    db_name: str = "cache.sqlite"

    def __post_init__(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(self.cache_dir) / self.db_name
        self._lock = threading.Lock()
        # ``check_same_thread=False`` lets one Cache instance serve threads
        # created after construction; the ``_lock`` serialises writes.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    updated   REAL NOT NULL DEFAULT (julianday('now')),
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_ns ON entries(namespace)"
            )
            self._conn.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Return the cached value for ``(namespace, key)`` or ``None``.

        Args:
            namespace: One of :data:`VALID_NAMESPACES`.
            key: Hex digest from :func:`make_key`.

        Returns:
            The previously stored value deserialised from JSON, or
            ``None`` if there is no entry.
        """
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")
        row = self._conn.execute(
            "SELECT value FROM entries WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store ``value`` for ``(namespace, key)``.

        Overwrites any previous entry atomically.

        Args:
            namespace: One of :data:`VALID_NAMESPACES`.
            key: Hex digest from :func:`make_key`.
            value: Any JSON-serialisable Python value.
        """
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entries(namespace, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    updated = julianday('now')
                """,
                (namespace, key, payload),
            )
            self._conn.commit()

    def delete(self, namespace: str, key: str) -> bool:
        """Delete ``(namespace, key)``. Returns whether a row was removed."""
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Wipe every entry under ``namespace``. Returns rows deleted."""
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE namespace = ?",
                (namespace,),
            )
            self._conn.commit()
            return cur.rowcount

    def contains(self, namespace: str, key: str) -> bool:
        """Return whether ``(namespace, key)`` is present."""
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE namespace = ? AND key = ? LIMIT 1",
            (namespace, key),
        ).fetchone()
        return row is not None

    def size(self, namespace: Optional[str] = None) -> int:
        """Count cached entries. If ``namespace`` is given, restrict to it."""
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("SELECT COUNT(*) FROM entries")
            else:
                if namespace not in VALID_NAMESPACES:
                    raise ValueError(f"unknown namespace: {namespace!r}")
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE namespace = ?",
                    (namespace,),
                )
            return int(cur.fetchone()[0])

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        with self._lock:
            self._conn.close()
            # Best-effort cleanup of WAL shadow files so the directory
            # is tidy when a caller wants to wipe state mid-test.
            for suffix in ("-wal", "-shm"):
                stray = Path(str(self._path) + suffix)
                if stray.exists():
                    try:
                        os.remove(stray)
                    except OSError:
                        pass
