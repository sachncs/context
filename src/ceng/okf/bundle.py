"""OKF bundles: a directory of concepts, read and written atomically."""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator

from ceng import errors
from ceng.okf import model


def validate_path(path: pathlib.PurePosixPath) -> pathlib.PurePosixPath:
    """Checks a bundle-relative concept path.

    Raises:
        ValidationError: If absolute, escaping, or not a `.md` file.
    """
    if path.is_absolute() or pathlib.PureWindowsPath(path).drive:
        raise errors.ValidationError(f"concept path must be relative: {path}")
    if any(part in ("..", "") for part in path.parts) or not path.parts:
        raise errors.ValidationError(f"concept path escapes bundle: {path}")
    if path.suffix != ".md":
        raise errors.ValidationError(f"concept path must end in .md: {path}")
    return path


@dataclasses.dataclass(frozen=True, slots=True)
class Bundle:
    """An immutable collection of placed concepts.

    Attributes:
        concepts: Concepts, each with a unique bundle-relative path.
    """

    concepts: tuple[model.Concept, ...] = ()

    def __post_init__(self) -> None:
        seen: set[pathlib.PurePosixPath] = set()
        for concept in self.concepts:
            if concept.path is None:
                raise errors.ValidationError(
                    "every concept in a bundle needs a path"
                )
            validate_path(concept.path)
            if concept.path in seen:
                raise errors.ValidationError(
                    f"duplicate concept path: {concept.path}"
                )
            seen.add(concept.path)

    def __iter__(self) -> Iterator[model.Concept]:
        return iter(self.concepts)

    def __len__(self) -> int:
        return len(self.concepts)

    def find(self, path: str | pathlib.PurePosixPath) -> model.Concept | None:
        """Returns the concept at `path`, or None."""
        wanted = pathlib.PurePosixPath(path)
        return next((c for c in self.concepts if c.path == wanted), None)

    def with_tag(self, tag: str) -> list[model.Concept]:
        """Returns concepts carrying `tag`, in bundle order."""
        return [c for c in self.concepts if tag in c.frontmatter.tags]

    def of_type(self, concept_type: str) -> list[model.Concept]:
        """Returns concepts of `concept_type`, in bundle order."""
        return [c for c in self.concepts if c.frontmatter.type == concept_type]

    def with_priority_at_least(self, minimum: int) -> list[model.Concept]:
        """Returns concepts at or above `minimum`, best first."""
        chosen = [c for c in self.concepts if c.frontmatter.priority >= minimum]
        return sorted(
            chosen,
            key=lambda c: (c.frontmatter.priority, c.frontmatter.helpful_count),
            reverse=True,
        )

    def write(self, directory: str | pathlib.Path) -> list[pathlib.Path]:
        """Writes the bundle to `directory` with an atomic swap.

        Files are staged in a sibling directory. The existing bundle is moved
        aside, the staging directory takes its place, and only then is the
        old copy deleted; on failure the old bundle is restored. Readers
        therefore never observe a partial bundle.

        Returns:
            Sorted absolute paths of the written files.

        Raises:
            ValidationError: If the target is a file.
        """
        root = pathlib.Path(directory).resolve()
        if root.exists() and not root.is_dir():
            raise errors.ValidationError(f"{root} is not a directory")
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix=f"{root.name}.staging-", dir=root.parent)
        )
        backup = root.with_name(f"{root.name}.old-{uuid.uuid4().hex[:8]}")
        moved = False
        try:
            for concept in self.concepts:
                if concept.path is None:  # unreachable: checked on construction
                    continue
                target = staging / concept.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(concept.render(), encoding="utf-8")
            if root.exists():
                root.replace(backup)
                moved = True
            staging.replace(root)
        except BaseException:
            if moved and not root.exists():
                backup.replace(root)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return sorted(root / c.path for c in self.concepts if c.path)

    @classmethod
    def read(cls, directory: str | pathlib.Path) -> Bundle:
        """Reads every `.md` file under `directory`.

        Raises:
            ValidationError: For invalid UTF-8, invalid concepts, or
                symlinks resolving outside the bundle.
        """
        root = pathlib.Path(directory)
        if not root.is_dir():
            raise errors.ValidationError(f"{root} is not a bundle directory")
        resolved_root = root.resolve()
        files = sorted(
            (p for p in root.rglob("*.md") if p.is_file()),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        concepts = []
        for file in files:
            if not file.resolve().is_relative_to(resolved_root):
                raise errors.ValidationError(
                    f"symlink target escapes bundle: {file}"
                )
            try:
                text = file.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise errors.ValidationError(
                    f"{file}: not valid UTF-8 ({exc})"
                ) from exc
            relative = pathlib.PurePosixPath(file.relative_to(root).as_posix())
            concepts.append(model.Concept.parse(text, relative))
        return cls(tuple(concepts))

    @classmethod
    def of(cls, concepts: Iterable[model.Concept]) -> Bundle:
        """Builds a bundle from any iterable of placed concepts."""
        return cls(tuple(concepts))
