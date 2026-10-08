"""Filesystem notes store; notes are OKF concepts on disk."""

from __future__ import annotations

import contextlib
import os
import pathlib
import threading

from foveate import errors
from foveate.internals import filelock
from foveate.okf import model
from foveate.stores import base

NOTE_TYPE = "foveate/note"
LOCK_NAME = ".store.lock"


@base.NotesStore.registry.register("filesystem")
class FilesystemNotesStore(base.NotesStore):
    """Stores each note as `<root>/<path>` with YAML frontmatter.

    Tags and timestamps persist in the frontmatter, writes are atomic
    (`os.replace` of a temporary file) and serialized across threads and
    processes by a lock file, which also makes `append` safe.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        """Opens (creating) the store directory.

        Raises:
            ConfigError: If the directory cannot be created.
        """
        self.root = pathlib.Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise errors.ConfigError(f"cannot create {self.root}") from exc
        self.root = self.root.resolve()
        self.thread_lock = threading.RLock()

    def locked(self) -> contextlib.AbstractContextManager[None]:
        """Returns the cross-process lock context manager."""
        return filelock.file_lock(self.root / LOCK_NAME)

    def resolve(self, path: str | pathlib.PurePosixPath) -> pathlib.Path:
        """Maps a note path to a file inside the root."""
        relative = base.validate_note_path(path)
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise errors.ValidationError(f"note path escapes store: {path}")
        return target

    def write(
        self,
        path: str | pathlib.PurePosixPath,
        content: str,
        tags: tuple[str, ...] = (),
    ) -> base.Note:
        with self.thread_lock, self.locked():
            return self.write_unlocked(path, content, tags)

    def write_unlocked(
        self,
        path: str | pathlib.PurePosixPath,
        content: str,
        tags: tuple[str, ...],
    ) -> base.Note:
        """Writes atomically; the caller must hold the locks."""
        target = self.resolve(path)
        note = base.Note(
            base.validate_note_path(path), content, tags, model.now_iso()
        )
        concept = model.Concept(
            model.Frontmatter(
                type=NOTE_TYPE, tags=tags, timestamp=note.timestamp
            ),
            content,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(concept.render(), encoding="utf-8")
        os.replace(temporary, target)
        return note

    def append(
        self, path: str | pathlib.PurePosixPath, content: str
    ) -> base.Note:
        with self.thread_lock, self.locked():
            try:
                existing = self.read(path)
            except errors.ValidationError:
                return self.write_unlocked(path, content, ())
            return self.write_unlocked(
                path, existing.content + content, existing.tags
            )

    def read(self, path: str | pathlib.PurePosixPath) -> base.Note:
        target = self.resolve(path)
        try:
            text = target.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            raise errors.ValidationError(f"no such note: {path}") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise errors.ValidationError(
                f"cannot read note {path}: {exc}"
            ) from exc
        concept = model.Concept.parse(text)
        return base.Note(
            base.validate_note_path(path),
            concept.body,
            concept.frontmatter.tags,
            concept.frontmatter.timestamp,
        )

    def delete(self, path: str | pathlib.PurePosixPath) -> bool:
        target = self.resolve(path)
        with self.thread_lock, self.locked():
            try:
                target.unlink()
            except FileNotFoundError:
                return False
        return True

    def entries(self, prefix: str = "") -> list[base.Note]:
        notes = []
        for file in sorted(self.root.rglob("*.md")):
            relative = file.relative_to(self.root).as_posix()
            if not relative.startswith(prefix):
                continue
            try:
                notes.append(self.read(relative))
            except errors.ValidationError:
                continue
        return sorted(notes, key=lambda n: (n.timestamp, str(n.path)))
