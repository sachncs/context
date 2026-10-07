"""Notes stores: durable scratch memory outside the context window."""

from __future__ import annotations

import abc
import dataclasses
import pathlib
import threading

from ceng import errors
from ceng.internals import registry

PINNED = "pinned"


@dataclasses.dataclass(frozen=True, slots=True)
class Note:
    """A stored note.

    Attributes:
        path: Store-relative POSIX path ending in `.md`.
        content: Markdown content.
        tags: Labels; the `pinned` tag exempts a note from eviction.
        timestamp: ISO 8601 time of the last write.
    """

    path: pathlib.PurePosixPath
    content: str
    tags: tuple[str, ...] = ()
    timestamp: str = ""

    @property
    def pinned(self) -> bool:
        """Returns whether the note is exempt from eviction."""
        return PINNED in self.tags

    @property
    def size(self) -> int:
        """Returns the content size in UTF-8 bytes."""
        return len(self.content.encode("utf-8"))


def validate_note_path(
    path: str | pathlib.PurePosixPath,
) -> pathlib.PurePosixPath:
    """Normalises and checks a note path.

    Raises:
        ValidationError: If absolute, escaping, or not a `.md` file.
    """
    candidate = pathlib.PurePosixPath(path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or not candidate.parts
    ):
        raise errors.ValidationError(f"invalid note path: {path}")
    if candidate.suffix != ".md":
        raise errors.ValidationError(f"note path must end in .md: {path}")
    return candidate


class NotesStore(abc.ABC):
    """Persistent key/value notes addressed by relative path."""

    registry: registry.Registry[type[NotesStore]] = registry.Registry(
        "notes store"
    )

    @abc.abstractmethod
    def write(
        self,
        path: str | pathlib.PurePosixPath,
        content: str,
        tags: tuple[str, ...] = (),
    ) -> Note:
        """Creates or replaces a note and returns it."""

    @abc.abstractmethod
    def read(self, path: str | pathlib.PurePosixPath) -> Note:
        """Returns the note at `path`.

        Raises:
            ValidationError: If the note does not exist or is malformed.
        """

    @abc.abstractmethod
    def delete(self, path: str | pathlib.PurePosixPath) -> bool:
        """Removes a note; returns whether it existed."""

    @abc.abstractmethod
    def entries(self, prefix: str = "") -> list[Note]:
        """Returns notes whose path starts with `prefix`, oldest first."""

    def append(self, path: str | pathlib.PurePosixPath, content: str) -> Note:
        """Appends `content` to a note (creating it), keeping its tags."""
        try:
            existing = self.read(path)
        except errors.ValidationError:
            return self.write(path, content)
        return self.write(path, existing.content + content, existing.tags)

    def total_size(self) -> int:
        """Returns the summed content bytes of all notes."""
        return sum(note.size for note in self.entries())

    def evict_to(self, max_bytes: int) -> list[pathlib.PurePosixPath]:
        """Deletes the oldest unpinned notes until the store fits.

        Args:
            max_bytes: Target total content size.

        Returns:
            Paths removed, oldest first. Pinned notes are never removed, so
            the store may remain above `max_bytes`.
        """
        notes = self.entries()
        total = sum(note.size for note in notes)
        removed = []
        for note in notes:
            if total <= max_bytes:
                break
            if note.pinned:
                continue
            self.delete(note.path)
            total -= note.size
            removed.append(note.path)
        return removed


@NotesStore.registry.register("memory")
class MemoryNotesStore(NotesStore):
    """Process-local store, useful for tests and ephemeral agents."""

    def __init__(self) -> None:
        self.notes: dict[pathlib.PurePosixPath, Note] = {}
        self.lock = threading.RLock()
        self.clock = 0

    def write(
        self,
        path: str | pathlib.PurePosixPath,
        content: str,
        tags: tuple[str, ...] = (),
    ) -> Note:
        key = validate_note_path(path)
        with self.lock:
            self.clock += 1
            note = Note(key, content, tags, f"{self.clock:012d}")
            self.notes[key] = note
            return note

    def read(self, path: str | pathlib.PurePosixPath) -> Note:
        with self.lock:
            try:
                return self.notes[validate_note_path(path)]
            except KeyError:
                raise errors.ValidationError(f"no such note: {path}") from None

    def delete(self, path: str | pathlib.PurePosixPath) -> bool:
        with self.lock:
            return self.notes.pop(validate_note_path(path), None) is not None

    def entries(self, prefix: str = "") -> list[Note]:
        with self.lock:
            chosen = [
                n for n in self.notes.values() if str(n.path).startswith(prefix)
            ]
        return sorted(chosen, key=lambda n: (n.timestamp, str(n.path)))
