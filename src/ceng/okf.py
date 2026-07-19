"""Open Knowledge Format (OKF) v0.1 — read, write, and bundle knowledge.

An OKF bundle is a directory of markdown files with YAML frontmatter.
Each file represents one *concept* — a unit of knowledge such as a
table, metric, runbook, or dataset. The file path inside the bundle
is the concept's identity, so cross-references are normal markdown
links like ``[orders](tables/orders.md)``.

This module provides:

* :class:`Frontmatter` — the small set of structured fields
  (``type``, ``title``, ``description``, ``resource``, ``tags``,
  ``timestamp``) that every OKF producer should know about.
* :class:`Concept` — a frontmatter + body pair, optionally paired
  with its path inside a bundle.
* :func:`render_concept` / :func:`parse_concept` — markdown
  serialisation with a hand-rolled YAML subset parser so the module
  stays dependency-free.
* :func:`read_concept_file` / :func:`write_concept_file` /
  :func:`read_bundle` / :func:`write_bundle` — file-level I/O for
  one concept or an entire directory at once.

Reserved filenames per the OKF v0.1 spec: ``index.md`` (progressive
disclosure of a subtree) and ``log.md`` (chronological history of
changes). Both are normal :class:`Concept` instances with their own
``type`` field; this module treats them as ordinary concept files.

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


OKF_VERSION = "0.1"
FRONTMATTER_OPEN = "---"
FRONTMATTER_CLOSE = "---"
RESERVED_INDEX = "index.md"
RESERVED_LOG = "log.md"

CENG_LEAF_SUMMARY = "ceng/leaf-summary"
CENG_COMBINED_SUMMARY = "ceng/combined-summary"
CENG_BUNDLE_INDEX = "ceng/bundle-index"


@dataclass(frozen=True)
class Frontmatter:
    """YAML frontmatter for an OKF concept.

    The OKF v0.1 spec mandates only ``type``. The other fields are
    conventional rather than required; :meth:`to_dict` omits empty
    values so a producer can pick and choose.

    Attributes:
        type: Required concept type, e.g. ``"BigQuery Table"`` or
            :data:`CENG_LEAF_SUMMARY`.
        title: Short, human-readable name.
        description: One- or two-sentence summary.
        resource: URL or IRI pointing at the underlying source.
        tags: Free-form list of categorisation strings.
        timestamp: ISO 8601 datetime string. Use
            :meth:`Frontmatter.now` to fill with the current UTC time.
        extra: Producer-defined additional fields. Stored as a tuple
            of ``(key, value)`` pairs so the dataclass stays
            hashable.
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
        """Construct a :class:`Frontmatter` from a dict.

        Unknown keys flow into ``extra`` so producers can carry
        domain-specific fields without this module growing.
        """
        kwargs: dict[str, Any] = {}
        extras: list[tuple[str, Any]] = []
        for key, value in d.items():
            if key == "type":
                kwargs["type"] = _coerce_str(value, key)
            elif key == "title":
                kwargs["title"] = _coerce_str(value, key)
            elif key == "description":
                kwargs["description"] = _coerce_str(value, key)
            elif key == "resource":
                kwargs["resource"] = _coerce_str(value, key)
            elif key == "tags":
                kwargs["tags"] = tuple(_coerce_str(v, "tags[]") for v in value)
            elif key == "timestamp":
                kwargs["timestamp"] = _coerce_str(value, key)
            else:
                extras.append((key, value))
        if "type" not in kwargs:
            raise ValueError("frontmatter is missing required 'type' field")
        kwargs["extra"] = tuple(extras)
        return cls(**kwargs)


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    A small convenience exported at module level so callers that
    build concepts incrementally don't have to construct a
    full :class:`Frontmatter` just for the timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Concept:
    """A single OKF concept: frontmatter, body, and (optional) bundle path.

    Attributes:
        frontmatter: The :class:`Frontmatter` describing the concept.
        body: Markdown body of the concept (after the frontmatter).
        path: Bundle-relative file path, e.g.
            ``"tables/orders.md"``. Optional until written. Strings
            and :class:`Path` objects are both accepted; the value
            is normalised to :class:`Path` on construction.
    """

    frontmatter: Frontmatter
    body: str = ""
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)

    def with_path(self, path: str | Path) -> "Concept":
        """Return a copy of this concept with ``path`` set."""
        copy = Concept(self.frontmatter, self.body, self.path)
        copy.path = Path(path)
        return copy


# ---------------------------------------------------------------------------
# YAML subset for OKF frontmatter
# ---------------------------------------------------------------------------


def render_frontmatter(d: dict[str, Any]) -> str:
    """Serialise ``d`` as a YAML block using the OKF subset grammar.

    Supported: scalars (str/int/float/bool/None), inline lists
    (``[a, b, c]``), nested mappings (one-key per line), block lists
    using ``-`` for entries. OKF v0.1 only requires scalars and inline
    lists, but supporting nested structures costs little.

    Args:
        d: Plain dict whose values are JSON-ish.

    Returns:
        YAML text (no surrounding ``---`` markers).
    """
    return _render_yaml(d, indent=0)


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse a YAML block produced by :func:`render_frontmatter`.

    Args:
        text: YAML text (no surrounding ``---`` markers).

    Returns:
        Parsed dict.

    Raises:
        ValueError: If the text can't be parsed.
    """
    return _parse_yaml(text)


def _render_yaml(value: Any, indent: int) -> str:
    """Recursive YAML renderer for the OKF subset."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for key, v in value.items():
            rendered = _render_yaml(v, indent + 1)
            if "\n" in rendered:
                lines.append(f"{pad}{key}:\n{rendered}")
            else:
                lines.append(f"{pad}{key}: {rendered}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
            return "[" + ", ".join(_render_yaml(v, indent) for v in value) + "]"
        out: list[str] = []
        for v in value:
            rendered = _render_yaml(v, indent + 1)
            head, _, rest = rendered.partition("\n")
            out.append(f"{pad}- {head}")
            if rest:
                out.append(rest)
        return "\n".join(out)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _quote_scalar(str(value))


def _quote_scalar(text: str) -> str:
    """Wrap a scalar in quotes if it would otherwise be ambiguous."""
    if text in {"", "null", "true", "false"} or text[0] in {"[", "]", "{", "}", "#", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"}:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if re.match(r"^[-+]?\d", text):
        return '"' + text + '"'
    return text


def _parse_yaml(text: str) -> dict[str, Any]:
    """Flat YAML parser for the OKF frontmatter subset.

    Supports ``key: scalar`` and ``key: [a, b, c]`` lines. Nested
    mappings are not part of the OKF v0.1 spec; this parser is
    deliberately minimal so it remains easy to audit.
    """
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip() != line:
            # OKF v0.1 frontmatter is flat; reject indented lines.
            raise ValueError(
                f"indented lines are not allowed in OKF frontmatter: {line!r}"
            )
        key, sep, value = line.partition(":")
        if sep == "":
            raise ValueError(f"not a key:value pair: {line!r}")
        out[key.strip()] = _parse_value(value.strip())
    return out


def _parse_value(text: str) -> Any:
    """Parse a YAML scalar or inline list."""
    if text.startswith("[") and text.endswith("]"):
        return _parse_inline_list(text)
    return _parse_scalar(text)


def _parse_inline_list(text: str) -> list[Any]:
    """Parse ``[a, b, c]``, respecting nested brackets and quoted strings."""
    inner = text[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(part.strip()) for part in _split_top_level(inner, ",")]


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` by ``sep`` ignoring nested brackets and quotes."""
    parts: list[str] = []
    depth = 0
    quote: Optional[str] = None
    buf: list[str] = []
    for char in text:
        if quote is not None:
            buf.append(char)
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            buf.append(char)
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(char)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_scalar(text: str) -> Any:
    """Parse a YAML scalar; returns int/float/bool/str/None."""
    if text == "" or text == "~" or text.lower() == "null":
        return None
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].encode("utf-8").decode("unicode_escape")
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _coerce_str(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"frontmatter field {field!r} must be a string, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Concept markdown serialisation
# ---------------------------------------------------------------------------


def render_concept(concept: Concept) -> str:
    """Render a :class:`Concept` as OKF markdown text.

    Args:
        concept: Concept to serialise.

    Returns:
        UTF-8 markdown text starting with the ``---`` frontmatter
        delimiter block, then a blank line, then the body.
    """
    d = concept.frontmatter.to_dict()
    yaml_block = render_frontmatter(d)
    body = concept.body or ""
    if body and not body.endswith("\n"):
        body = body + "\n"
    return f"{FRONTMATTER_OPEN}\n{yaml_block}\n{FRONTMATTER_CLOSE}\n\n{body}"


def parse_concept(text: str, path: Optional[Path] = None) -> Concept:
    """Parse OKF markdown into a :class:`Concept`.

    Args:
        text: Full markdown text including frontmatter.
        path: Optional bundle-relative path to record on the result
            (string or :class:`Path`).

    Returns:
        The parsed :class:`Concept`. ``path`` is always normalised
        to :class:`Path`. Leading and trailing newlines around the
        body are stripped so the round-trip is idempotent.

    Raises:
        ValueError: If the text doesn't start with a YAML frontmatter
            block delimited by ``---``.
    """
    if not text.startswith(f"{FRONTMATTER_OPEN}\n"):
        raise ValueError(
            "OKF concept text must begin with '---\\n' followed by YAML frontmatter"
        )
    after_open = text[len(FRONTMATTER_OPEN) + 1 :]
    close_idx = after_open.find(f"\n{FRONTMATTER_CLOSE}\n")
    if close_idx == -1:
        raise ValueError("frontmatter is not terminated by '---' on its own line")
    yaml_text = after_open[:close_idx]
    body = after_open[close_idx + len(FRONTMATTER_CLOSE) + 2 :]
    frontmatter_dict = parse_frontmatter(yaml_text)
    frontmatter = Frontmatter.from_dict(frontmatter_dict)
    normalised_path = Path(path) if path is not None else None
    stripped_body = body.strip("\n")
    return Concept(
        frontmatter=frontmatter,
        body=stripped_body,
        path=normalised_path,
    )


# ---------------------------------------------------------------------------
# File-level I/O
# ---------------------------------------------------------------------------


def read_concept_file(path: str | Path) -> Concept:
    """Read one OKF concept from disk.

    Args:
        path: Path to a ``.md`` file.

    Returns:
        The parsed :class:`Concept` with ``path`` set.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    parsed = parse_concept(text, path=p)
    return parsed


def write_concept_file(path: str | Path, concept: Concept) -> None:
    """Write one OKF concept to disk, creating parent directories.

    Args:
        path: Destination file. Should end in ``.md``.
        concept: The :class:`Concept` to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_concept(concept), encoding="utf-8")


def write_bundle(
    directory: str | Path, concepts: Iterable[Concept]
) -> list[Path]:
    """Write an OKF bundle: each concept as one ``.md`` file.

    Args:
        directory: Destination directory. Created on demand.
        concepts: Iterable of :class:`Concept` instances. Every
            concept must have a ``path`` set; the path is interpreted
            relative to ``directory``.

    Returns:
        The list of file paths written, sorted.

    Raises:
        ValueError: If any concept has no ``path`` set, or its path
            would escape the bundle root, or the file is not a
            ``.md`` file.
    """
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for concept in concepts:
        if concept.path is None:
            raise ValueError("every concept in a bundle must have a path set")
        rel = Path(concept.path)
        if rel.is_absolute():
            raise ValueError(f"concept path must be bundle-relative, got {rel}")
        parts = rel.parts
        if any(part in {"..", ""} for part in parts):
            raise ValueError(f"concept path escapes bundle root: {rel}")
        if rel.suffix != ".md":
            raise ValueError(f"concept path must end in .md, got {rel}")
        target = (root / rel).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"concept path escapes bundle root: {rel}")
        write_concept_file(target, concept)
        written.append(target)
    return sorted(written)


def read_bundle(directory: str | Path) -> list[Concept]:
    """Read every ``.md`` file under ``directory`` as an OKF concept.

    The directory is walked recursively. Non-markdown files are
    skipped silently; malformed frontmatter raises
    :class:`ValueError`.

    Args:
        directory: Bundle directory.

    Returns:
        A list of :class:`Concept` instances sorted by their bundle-
        relative path.

    Raises:
        ValueError: If a ``.md`` file has malformed frontmatter.
    """
    root = Path(directory)
    if not root.exists():
        return []
    abs_paths = sorted(
        (p for p in root.rglob("*.md") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    out: list[Concept] = []
    for abs_path in abs_paths:
        rel = abs_path.relative_to(root).as_posix()
        text = abs_path.read_text(encoding="utf-8")
        out.append(parse_concept(text, path=Path(rel)))
    return out


def find_concept(
    bundle: list[Concept], path: str | Path
) -> Optional[Concept]:
    """Find a concept by bundle-relative path within ``bundle``.

    Args:
        bundle: Iterable of :class:`Concept` (e.g. from
            :func:`read_bundle`).
        path: Bundle-relative path (forward-slash or os-native).

    Returns:
        The matching :class:`Concept` or ``None``.
    """
    needle = Path(path).as_posix()
    for concept in bundle:
        if concept.path is None:
            continue
        if concept.path.as_posix() == needle:
            return concept
    return None


def cross_links(body: str) -> list[str]:
    """Extract OKF cross-link targets from a markdown body.

    A cross-link is a markdown link whose target ends in ``.md``.
    Returns bundle-relative paths.

    Args:
        body: Markdown body text.

    Returns:
        List of bundle-relative path strings, in document order.
    """
    return [match.group(1) for match in re.finditer(r"\[[^\]]*\]\(([^)]+\.md)\)", body)]
