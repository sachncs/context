"""sqlite-backed cache for LLM call results.

The cache stores serialised values under deterministic keys. Two
namespaces keep concerns separate:

* ``"summarize"`` — leaf-context summaries produced by
  :mod:`ceng.compress`.
* ``"ppa_check"`` — leaf answers produced by :mod:`ceng.check`.

The cache uses only the Python standard library. Key derivation uses
SHA-256 over a canonical JSON dump, so two callers producing the
same arguments hit the same row without having to serialise the value
first.

Typical use::

    with Cache(cache_dir=".ceng_cache") as cache:
        cache.set("summarize", key="abc123", value={"summary": "..."})
        hit = cache.get("summarize", key="abc123")  # {"summary": "..."}
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1"
NAMESPACE_SUMMARIZE = "summarize"
NAMESPACE_PPA_CHECK = "ppa_check"

VALID_NAMESPACES = frozenset({NAMESPACE_SUMMARIZE, NAMESPACE_PPA_CHECK})

DEFAULT_CACHE_ROOT = ".ceng/cache"


def make_key(parts: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for a dict of key parts.

    Args:
        parts: Mapping whose values are JSON-serialisable.

    Returns:
        64-character hex digest usable as a cache key.

    Raises:
        TypeError: If any value is not JSON-serialisable.
    """
    canonical = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Cache:
    """On-disk key/value cache backed by sqlite.

    Thread-safe. One process owns the file. For multi-process
    sharing, point multiple :class:`Cache` instances at the same
    ``cache_dir``; SQLite's WAL mode plus a busy-timeout default of
    five seconds handles cross-process serialisation.

    Use as a context manager::

        with Cache(cache_dir=".ceng_cache") as cache:
            cache.set("summarize", key="k", value={"x": 1})

    Attributes:
        cache_dir: Directory the sqlite file lives in. Created on demand.
        db_name: Filename of the sqlite database inside ``cache_dir``.
    """

    cache_dir: str = DEFAULT_CACHE_ROOT
    db_name: str = "cache.sqlite"
    busy_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.path: Path = Path(self.cache_dir) / self.db_name
        self.lock = threading.Lock()
        # ``check_same_thread=False`` lets one Cache instance serve
        # threads created after construction; the lock serialises
        # every connection operation below.
        self.conn: sqlite3.Connection = sqlite3.connect(
            self.path, check_same_thread=False
        )
        self.conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.lock:
            self._ensure_schema()
            self._enforce_schema_version()

    def __enter__(self) -> "Cache":
        """Return self for use in ``with`` statements."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the cache on context-manager exit."""
        self.close()

    def _ensure_schema(self) -> None:
        """Create the entries table and namespace index if absent."""
        self.conn.execute(
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
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_ns ON entries(namespace)"
        )
        self.conn.commit()

    def _enforce_schema_version(self) -> None:
        """Check ``PRAGMA user_version`` matches :data:`SCHEMA_VERSION`.

        An older or unknown version on disk raises :class:`RuntimeError`
        so a silent corruption from a downgrade is loud. Set
        ``user_version`` to the current value when migrating forward.
        """
        row = self.conn.execute("PRAGMA user_version").fetchone()
        on_disk = int(row[0]) if row else 0
        if on_disk == 0:
            self.conn.execute(
                f"PRAGMA user_version = {int(SCHEMA_VERSION)}"
            )
            self.conn.commit()
        elif on_disk != int(SCHEMA_VERSION):
            raise RuntimeError(
                f"cache schema version mismatch: file={on_disk} "
                f"expected={SCHEMA_VERSION}; clear the cache directory "
                f"or migrate manually"
            )

    def _check_namespace(self, namespace: str) -> None:
        if namespace not in VALID_NAMESPACES:
            raise ValueError(f"unknown namespace: {namespace!r}")

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Return the cached value for ``(namespace, key)`` or ``None``.

        Args:
            namespace: One of :data:`VALID_NAMESPACES`.
            key: Hex digest from :func:`make_key`.

        Returns:
            The previously stored value deserialised from JSON, or
            ``None`` if there is no entry.
        """
        self._check_namespace(namespace)
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store ``value`` for ``(namespace, key)``.

        Overwrites any previous entry atomically. ``value`` must be
        JSON-serialisable and must not be ``None`` (which is the miss
        sentinel).

        Args:
            namespace: One of :data:`VALID_NAMESPACES`.
            key: Hex digest from :func:`make_key`.
            value: A JSON-serialisable Python value other than ``None``.
        """
        if value is None:
            raise ValueError("cache values cannot be None (use contains() to disambiguate misses)")
        self._check_namespace(namespace)
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO entries(namespace, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        value = excluded.value,
                        updated = julianday('now')
                    """,
                    (namespace, key, payload),
                )
                self.conn.commit()
            except sqlite3.Error:
                self.conn.rollback()
                raise

    def delete(self, namespace: str, key: str) -> bool:
        """Delete ``(namespace, key)``. Returns whether a row was removed."""
        self._check_namespace(namespace)
        with self.lock:
            try:
                cur = self.conn.execute(
                    "DELETE FROM entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                self.conn.commit()
                return cur.rowcount > 0
            except sqlite3.Error:
                self.conn.rollback()
                raise

    def clear_namespace(self, namespace: str) -> int:
        """Wipe every entry under ``namespace``. Returns rows deleted."""
        self._check_namespace(namespace)
        with self.lock:
            try:
                cur = self.conn.execute(
                    "DELETE FROM entries WHERE namespace = ?",
                    (namespace,),
                )
                self.conn.commit()
                return int(cur.rowcount)
            except sqlite3.Error:
                self.conn.rollback()
                raise

    def contains(self, namespace: str, key: str) -> bool:
        """Return whether ``(namespace, key)`` is present."""
        self._check_namespace(namespace)
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM entries WHERE namespace = ? AND key = ? LIMIT 1",
                (namespace, key),
            ).fetchone()
        return row is not None

    def size(self, namespace: Optional[str] = None) -> int:
        """Count cached entries. If ``namespace`` is given, restrict to it."""
        if namespace is not None:
            self._check_namespace(namespace)
            sql = "SELECT COUNT(*) FROM entries WHERE namespace = ?"
            params: tuple[Any, ...] = (namespace,)
        else:
            sql = "SELECT COUNT(*) FROM entries"
            params = ()
        with self.lock:
            cur = self.conn.execute(sql, params)
            return int(cur.fetchone()[0])

    def close(self) -> None:
        """Close the underlying sqlite connection.

        Issues a ``wal_checkpoint(TRUNCATE)`` first so the on-disk
        WAL file is shrunk rather than left to grow in long-running
        processes. Safe to call multiple times.
        """
        with self.lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                # If the checkpoint fails (e.g. another process holds
                # the WAL), we still want to close cleanly.
                pass
            self.conn.close()
