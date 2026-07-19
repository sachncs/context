"""Open Knowledge Format (OKF) v0.1 reader and writer.

The format: a directory of markdown files with YAML frontmatter. One
concept = one file. The file path inside the bundle is the concept's
identity; cross-references are normal ``[text](path/to/concept.md)``
markdown links.

OKF v0.1 mandates only the ``type`` frontmatter field. ``title``,
``description``, ``resource``, ``tags``, and ``timestamp`` are
conventional. Anything else flows into ``Frontmatter.extra`` and must
itself be OKF-acceptable (scalars and lists of scalars); nested
mappings belong in the body.

This module provides:

* :class:`Frontmatter` and :class:`Concept`
* :func:`render_frontmatter` / :func:`parse_frontmatter`
  (delegated to PyYAML)
* :func:`render_concept` / :func:`parse_concept`
* :func:`read_concept_file` / :func:`write_concept_file`
* :func:`read_bundle` / :func:`write_bundle`
* :func:`find_concept` / :func:`cross_links`
* :func:`now_iso` for timestamp defaults
* Module-level constants for type names and reserved filenames

References:

* https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
* https://arxiv.org/abs/2510.26493 (Context Engineering 2.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


OKF_VERSION = "0.1"
RESERVED_INDEX = "index.md"
RESERVED_LOG = "log.md"

CENG_LEAF_SUMMARY = "ceng/leaf-summary"
CENG_COMBINED_SUMMARY = "ceng/combined-summary"
CENG_BUNDLE_INDEX = "ceng/bundle-index"


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def is_okf_scalar(value: Any) -> bool:
    """Return whether ``value`` is acceptable as an OKF frontmatter scalar.

    Scalars are ``str``, ``int``, ``float``, ``bool``, ``None``, or a
    list whose every element is itself a scalar. Anything else (nested
    mappings, sets, dates, dataclasses, ...) is rejected.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_okf_scalar(item) for item in value)
    return False


def coerce_str(value: Any, field: str) -> str:
    """Coerce ``value`` to ``str`` or raise :class:`ValueError`."""
    if isinstance(value, str):
        return value
    raise ValueError(
        f"frontmatter field {field!r} must be a string, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class Frontmatter:
    """YAML frontmatter for an OKF concept.

    OKF v0.1 mandates only ``type``. The remaining fields are
    conventional. ``extra`` holds producer-defined scalar fields and
    is rejected at construction time for any non-OKF-scalar value
    (see :func:`is_okf_scalar`).
    """

    type: str
    title: str = ""
    description: str = ""
    resource: str = ""
    tags: tuple[str, ...] = ()
    timestamp: str = ""
    extra: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def now(cls, **kw: Any) -> "Frontmatter":
        """Return a :class:`Frontmatter` with ``timestamp`` set to UTC now."""
        return cls(timestamp=now_iso(), **kw)

    def to_dict(self) -> dict[str, Any]:
        """Render the frontmatter as a plain dict, omitting empty fields."""
        out: dict[str, Any] = {"type": self.type}
        if self.title:
            out["title"] = self.title
        if self.description:
            out["description"] = self.description
        if self.resource:
            out["resource"] = self.resource
        if self.tags:
            out["tags"] = list(self.tags)
        if self.timestamp:
            out["timestamp"] = self.timestamp
        for key, value in self.extra:
            out[key] = value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Frontmatter":
        """Build a :class:`Frontmatter` from a dict.

        Unknown keys flow into ``extra`` and must be OKF-acceptable
        scalars or lists of scalars; raises :class:`ValueError`
        otherwise.
        """
        known = {"type", "title", "description", "resource", "tags", "timestamp"}
        kwargs: dict[str, Any] = {}
        extras: list[tuple[str, Any]] = []
        for key, value in d.items():
            if key == "type":
                kwargs["type"] = coerce_str(value, key)
            elif key == "title":
                kwargs["title"] = coerce_str(value, key)
            elif key == "description":
                kwargs["description"] = coerce_str(value, key)
            elif key == "resource":
                kwargs["resource"] = coerce_str(value, key)
            elif key == "tags":
                kwargs["tags"] = tuple(coerce_str(v, "tags[]") for v in value)
            elif key == "timestamp":
                kwargs["timestamp"] = coerce_str(value, key)
            elif key in known:
                pass
            else:
                if not is_okf_scalar(value):
                    raise ValueError(
                        f"frontmatter field {key!r} must be a scalar or list of "
                        f"scalars, got {type(value).__name__}"
                    )
                extras.append((key, value))
        if "type" not in kwargs:
            raise ValueError("frontmatter is missing required 'type' field")
        kwargs["extra"] = tuple(extras)
        return cls(**kwargs)


@dataclass
class Concept:
    """A single OKF concept: frontmatter, body, and (optional) bundle path.

    ``path`` is normalised to :class:`Path` at construction time. When
    a concept is added to a bundle (:func:`write_bundle`) the path is
    validated as bundle-relative; bare construction accepts any path
    because round-tripping :func:`read_concept_file` puts the
    absolute on-disk path there too.
    """

    frontmatter: Frontmatter
    body: str = ""
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, Path):
            self.path = Path(self.path)

    def with_path(self, path: str | Path) -> "Concept":
        """Return a copy of this concept with ``path`` set."""
        return Concept(self.frontmatter, self.body, Path(path))


def validate_bundle_path(rel: Path) -> Path:
    """Validate a bundle-relative concept path: no escape, must end in ``.md``.

    Raises:
        ValueError: If ``rel`` is absolute, escapes the bundle root,
            or has the wrong suffix.
    """
    if rel.is_absolute():
        raise ValueError(f"concept path must be bundle-relative, got {rel}")
    if any(part in {"..", ""} for part in rel.parts):
        raise ValueError(f"concept path escapes bundle root: {rel}")
    if rel.suffix != ".md":
        raise ValueError(f"concept path must end in .md, got {rel}")
    return rel


def render_frontmatter(d: dict[str, Any]) -> str:
    """Serialise ``d`` as YAML text (no surrounding ``---`` markers)."""
    return yaml.safe_dump(
        d, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip("\n")


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse a YAML block. Raises :class:`ValueError` on any parse error.

    Returns ``{}`` for an empty / whitespace-only block.
    """
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"frontmatter must be a mapping, got {type(loaded).__name__}"
        )
    return loaded


def render_concept(concept: Concept) -> str:
    """Render a :class:`Concept` as OKF markdown text."""
    yaml_block = render_frontmatter(concept.frontmatter.to_dict())
    body = concept.body or ""
    if body and not body.endswith("\n"):
        body = body + "\n"
    return f"---\n{yaml_block}\n---\n\n{body}"


def parse_concept(text: str, path: Optional[Path] = None) -> Concept:
    """Parse OKF markdown into a :class:`Concept`.

    Normalises line endings (CRLF and bare CR collapse to LF) and
    strips a leading UTF-8 BOM so the frontmatter lookup works on
    files authored on Windows or with editors that emit BOMs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    if not text.startswith("---\n"):
        raise ValueError(
            "OKF concept text must begin with '---\\n' followed by YAML frontmatter"
        )
    after_open = text[4:]
    close_idx = after_open.find("\n---\n")
    if close_idx == -1:
        raise ValueError("frontmatter is not terminated by '---' on its own line")
    yaml_text = after_open[:close_idx]
    body = after_open[close_idx + 5 :].strip("\n")
    return Concept(
        frontmatter=Frontmatter.from_dict(parse_frontmatter(yaml_text)),
        body=body or "",
        path=path,
    )


def read_concept_file(path: str | Path) -> Concept:
    """Read one OKF concept from disk.

    Raises :class:`ValueError` with the offending file path if the
    file is not valid UTF-8.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{p}: not valid UTF-8 ({exc})") from exc
    return parse_concept(text, path=p)


def write_concept_file(path: str | Path, concept: Concept) -> None:
    """Write one OKF concept to disk, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_concept(concept), encoding="utf-8")


def write_bundle(directory: str | Path, concepts: Iterable[Concept]) -> list[Path]:
    """Write an OKF bundle to ``directory``. One ``.md`` per concept.

    Writes happen to a sibling staging directory first, then the
    staging directory is renamed to ``directory`` so a crash mid-
    write never leaves a partially-written bundle visible to
    readers (POSIX rename is atomic; on Windows the operation is
    best-effort).

    Args:
        directory: Destination directory. Created on demand.
        concepts: Iterable of :class:`Concept`. Every concept must
            have a ``path`` set; duplicate paths raise.

    Returns:
        The list of written file paths, sorted.

    Raises:
        ValueError: If a concept has no ``path``, or its path is
            unsafe (absolute, escapes the bundle, wrong suffix, or
            duplicated).
    """
    import shutil
    import tempfile

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=root.name + ".staging-", dir=str(root.parent)))
    written: list[Path] = []
    seen: set[Path] = set()
    try:
        for concept in concepts:
            if concept.path is None:
                raise ValueError(
                    "every concept in a bundle must have a path set"
                )
            rel = validate_bundle_path(concept.path)
            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"concept path escapes bundle root: {rel}")
            if target in seen:
                raise ValueError(f"duplicate concept path: {rel}")
            seen.add(target)
            staging_target = staging / rel
            write_concept_file(staging_target, concept)
            written.append(target)
        # All writes succeeded: atomic swap.
        shutil.rmtree(root, ignore_errors=True)
        staging.replace(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return sorted(written)


def read_bundle(directory: str | Path) -> list[Concept]:
    """Read every ``.md`` under ``directory`` as an OKF concept.

    Each returned :class:`Concept` carries its bundle-relative
    ``path`` (forward-slash form), so :func:`find_concept` and
    cross-link lookups behave the same way as for hand-built bundles.

    Symlinks whose target resolves outside the bundle root are
    refused with :class:`ValueError` so an attacker cannot read
    arbitrary files by planting a symlink in the bundle.
    """
    root = Path(directory)
    if not root.exists():
        return []
    abs_root = root.resolve()
    abs_paths = sorted(
        (p for p in root.rglob("*.md") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    out: list[Concept] = []
    for abs_path in abs_paths:
        try:
            abs_path.resolve().relative_to(abs_root)
        except ValueError as exc:
            raise ValueError(
                f"symlink target escapes bundle: {abs_path}"
            ) from exc
        rel = abs_path.relative_to(root).as_posix()
        try:
            text = abs_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{abs_path}: not valid UTF-8 ({exc})") from exc
        out.append(parse_concept(text, path=Path(rel)))
    return out


def find_concept(
    bundle: Iterable[Concept], path: str | Path
) -> Optional[Concept]:
    """Find a concept by bundle-relative path."""
    needle = Path(path).as_posix()
    for concept in bundle:
        if concept.path is None:
            continue
        if concept.path.as_posix() == needle:
            return concept
    return None


_MD_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]*\]\(([^)\s]+?\.md)(?:\s+\"[^\"]*\")?\)"
)


def cross_links(body: str) -> list[str]:
    """Return bundle-relative ``.md`` cross-link targets in document order.

    External URLs and image syntax are filtered out.
    """
    out: list[str] = []
    for match in _MD_LINK_RE.finditer(body):
        target = match.group(1)
        if "://" in target:
            continue
        out.append(target)
    return out
