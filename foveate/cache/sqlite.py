"""SQLite-backed persistent cache."""

from __future__ import annotations

import contextlib
import logging
import pathlib
import sqlite3
import threading
import time

from foveate import errors
from foveate.cache import base

logger = logging.getLogger("foveate.cache")

SCHEMA_VERSION = 2


@base.Cache.registry.register("sqlite")
class SqliteCache(base.Cache):
    """Persistent cache with TTL and size-bounded eviction.

    A corrupted database file is quarantined (renamed with a `.corrupt`
    suffix) and recreated rather than failing the caller; read errors are
    treated as misses.
    """

    def __init__(
        self,
        directory: str | pathlib.Path,
        *,
        filename: str = "cache.db",
        ttl_seconds: float | None = None,
        max_entries: int = 100_000,
        busy_timeout_seconds: float = 30.0,
    ) -> None:
        """Opens (creating if needed) the cache database.

        Args:
            directory: Folder holding the database file.
            filename: Database file name.
            ttl_seconds: Entry lifetime, or None for no expiry.
            max_entries: Oldest entries beyond this count are evicted.
            busy_timeout_seconds: SQLite lock wait.

        Raises:
            ConfigError: If limits are invalid or the directory is unusable.
        """
        if max_entries < 1:
            raise errors.ConfigError("max_entries must be >= 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise errors.ConfigError("ttl_seconds must be positive")
        self.path = pathlib.Path(directory) / filename
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.busy_timeout_seconds = busy_timeout_seconds
        self.lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise errors.ConfigError(
                f"cannot create {self.path.parent}"
            ) from exc
        self.connection = self.open()

    def open(self) -> sqlite3.Connection:
        """Connects, quarantining the file once if it is corrupt."""
        try:
            return self.connect()
        except sqlite3.DatabaseError:
            logger.warning("cache %s is corrupt; recreating", self.path)
            with contextlib.suppress(OSError):
                self.path.replace(self.path.with_suffix(".corrupt"))
            return self.connect()

    def connect(self) -> sqlite3.Connection:
        """Opens a connection and ensures the schema is current."""
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_seconds,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                connection.execute("DROP TABLE IF EXISTS entries")
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "stored REAL NOT NULL)"
            )
            connection.commit()
        except sqlite3.DatabaseError:
            connection.close()
            raise
        return connection

    def get(self, key: str) -> str | None:
        with self.lock:
            try:
                row = self.connection.execute(
                    "SELECT value, stored FROM entries WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.DatabaseError:
                logger.warning("cache read failed for %s", key, exc_info=True)
                return None
        if row is None:
            return None
        value, stored = row
        if (
            self.ttl_seconds is not None
            and time.time() - stored > self.ttl_seconds
        ):
            self.delete(key)
            return None
        return str(value)

    def set(self, key: str, value: str) -> None:
        with self.lock:
            try:
                self.connection.execute(
                    "INSERT OR REPLACE INTO entries (key, value, stored) "
                    "VALUES (?, ?, ?)",
                    (key, value, time.time()),
                )
                self.connection.execute(
                    "DELETE FROM entries WHERE key IN (SELECT key FROM entries "
                    "ORDER BY stored DESC LIMIT -1 OFFSET ?)",
                    (self.max_entries,),
                )
                self.connection.commit()
            except sqlite3.DatabaseError:
                logger.warning("cache write failed for %s", key, exc_info=True)

    def delete(self, key: str) -> None:
        with self.lock, contextlib.suppress(sqlite3.DatabaseError):
            self.connection.execute("DELETE FROM entries WHERE key = ?", (key,))
            self.connection.commit()

    def clear(self) -> None:
        with self.lock, contextlib.suppress(sqlite3.DatabaseError):
            self.connection.execute("DELETE FROM entries")
            self.connection.commit()

    def close(self) -> None:
        with self.lock, contextlib.suppress(sqlite3.Error):
            self.connection.close()
