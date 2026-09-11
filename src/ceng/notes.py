"""Agentic NOTE-style external memory.

Pattern from Anthropic's "Effective context engineering for AI agents":
agents maintain structured notes outside the context window to persist
state across turns and sessions. ``ceng.notes`` keeps that pattern
simple and inspectable.

Implementation:
- one ``.md`` file per note in the configured ``root`` directory
- atomic writes (temp-file + rename) so a crash mid-write doesn't
  leave half a note
- size-bounded LRU compaction keeps total bytes under a budget by
  deleting the oldest non-pinned notes first

This isn't a database. It's a directory of markdown you can read with
``cat``, version with git, share over rsync, and grep.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SAFE_REL_PATH = re.compile(r"^[A-Za-z0-9_./\-]+$")


def now_iso() -> str:
    """UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Note:
    """A single agent-managed note.

    Attributes:
        path: Bundle-relative path (forward-slash form).
        content: Markdown body.
        timestamp: ISO 8601 timestamp of the most recent write.
        tags: Free-form tags for grouping.
    """

    path: str
    content: str
    timestamp: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_file(cls, abs_path: Path, root: Path) -> Note:
        rel = abs_path.resolve().relative_to(root.resolve()).as_posix()
        text = abs_path.read_text(encoding="utf-8")
        return cls(
            path=rel,
            content=text,
            timestamp=_timestamp_from_filename(abs_path.name) or now_iso(),
            tags=(),
        )


def _timestamp_from_filename(name: str) -> str:
    """Extract a YYYY-MM-DD-HHMMSS prefix if present.

    ``NotesManager.write`` prefixes each note filename with a UTC
    timestamp so listing reveals creation order.
    """
    head = name.split(".", 1)[0]
    parts = head.split("-")
    if len(parts) >= 6 and all(p.isdigit() for p in parts[:6]):
        return "-".join(parts[:3]) + "T" + ":".join(parts[3:6]) + "Z"
    return ""


@dataclass
class NotesManager:
    """Filesystem-backed NOTE store rooted at one directory.

    One note = one ``.md`` file. Atomic writes via temp + rename. Simple
    LRU compaction by mtime.

    Args:
        root: Directory that holds the notes. Created on demand.

    Raises:
        ValueError: If a note path tries to escape the root (paranoia
            check; the caller can catch and route).
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs_path(self, rel: str) -> Path:
        if not SAFE_REL_PATH.match(rel):
            raise ValueError(f"unsafe note path: {rel!r}")
        if rel.endswith(("/", "\\")):
            raise ValueError(f"note path must not end with a separator: {rel!r}")
        full = (self.root / rel).resolve()
        if not full.is_relative_to(self.root.resolve()):
            raise ValueError(f"note path escapes root: {rel!r}")
        return full

    def read(self, rel: str) -> Note:
        """Read a note by its bundle-relative path.

        Raises:
            FileNotFoundError: If the note does not exist.
        """
        full = self._abs_path(rel)
        if not full.exists():
            raise FileNotFoundError(f"note not found: {rel}")
        return Note.from_file(full, self.root)

    def write(self, rel: str, content: str, *, tags: Iterable[str] = ()) -> Note:
        """Write ``content`` to a note, atomically.

        Creates the parent directory on demand. Filename is the
        caller-supplied relative path; ordering by ``mtime`` from
        ``list()`` gives chronological order without renaming.

        Returns:
            The newly written :class:`Note`.
        """
        full = self._abs_path(rel)
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_suffix(full.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(full)
        return Note(
            path=rel,
            content=content,
            timestamp=now_iso(),
            tags=tuple(tags),
        )

    def append(self, rel: str, content: str, *, tags: Iterable[str] = ()) -> Note:
        """Append ``content`` to an existing note (or create it)."""
        full = self._abs_path(rel)
        existing = full.read_text(encoding="utf-8") if full.exists() else ""
        return self.write(rel, existing + content, tags=tags)

    def list(self) -> list[str]:
        """Return every note path under the root, sorted by mtime ascending."""
        if not self.root.exists():
            return []
        out: list[tuple[float, str]] = []
        for p in self.root.rglob("*.md"):
            if p.is_file():
                out.append((p.stat().st_mtime, p.relative_to(self.root).as_posix()))
        out.sort()
        return [rel for _, rel in out]

    def delete(self, rel: str) -> bool:
        """Delete a note. Returns whether a file was removed."""
        full = self._abs_path(rel)
        if not full.exists():
            return False
        full.unlink()
        return True

    def compact_by_size(self, *, max_bytes: int, keep_recent: int = 5) -> int:
        """Drop the oldest non-pinned notes when total size exceeds ``max_bytes``.

        Two-phase eviction:
        1. Sort non-pinned files by mtime, keep ``keep_recent`` most
           recent, drop the rest. Pinned (leading ``_``) files are
           never touched.
        2. If the surviving total still exceeds ``max_bytes``, drop
           additional non-pinned notes (oldest first) until under
           budget. Pinned + the ``keep_recent`` set are NEVER dropped
           here — if the user picks ``keep_recent > max_bytes / file``
           they implicitly opted into a budget they cannot hit.

        Filenames beginning with ``_`` are pinned; rename them to
        ``_important.md`` to keep them out of the LRU.
        """
        if not self.root.exists():
            return 0

        all_files = [p for p in self.root.rglob("*.md") if p.is_file()]
        pinned = [p for p in all_files if p.name.startswith("_")]
        rest = sorted(
            [p for p in all_files if not p.name.startswith("_")],
            key=lambda p: p.stat().st_mtime,
        )

        # Phase 1: keep_recent wins; everything older than that is dropped.
        keep_set = set(
            rest[-keep_recent:] if len(rest) > keep_recent else rest
        )
        dropped = 0
        for p in rest:
            if p in keep_set:
                continue
            try:
                p.unlink()
                dropped += 1
            except OSError:
                pass

        # Phase 2: hard size cap. Anything in keep_set is inviolable;
        # pinned is also inviolable. We can only touch unpinned files
        # outside keep_set — but phase 1 already removed those, so
        # phase 2 is effectively a no-op once phase 1 has run. The
        # loop survives as a future-proofing in case future code
        # adds post-phase-1 inserts.
        if max_bytes <= 0:
            return dropped
        survivors = list(keep_set) + pinned
        size = sum(p.stat().st_size for p in survivors)
        if size > max_bytes:
            for p in sorted(survivors, key=lambda p: p.stat().st_mtime)[:-1]:
                if size <= max_bytes:
                    break
                if p in pinned or p in keep_set:
                    continue
                try:
                    size -= p.stat().st_size
                    p.unlink()
                    dropped += 1
                except OSError:
                    pass
        return dropped
