"""OKF (Open Knowledge Format) v0.1 data model and markdown rendering.

A concept is one markdown file with YAML frontmatter; its bundle-relative
path is its identity and cross-references are ordinary markdown links.
Only `type` is mandatory. Unknown frontmatter keys flow into `extra` and
must be scalars or lists of scalars (nested data belongs in the body).

References:
    https://arxiv.org/abs/2510.26493
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import yaml

from ceng import errors

OKF_VERSION = "0.1"
RESERVED_INDEX = "index.md"
RESERVED_LOG = "log.md"

STRING_FIELDS = ("title", "description", "resource", "timestamp", "expires_at")
INTEGER_FIELDS = ("priority", "helpful_count", "harmful_count")
KNOWN_FIELDS = frozenset(("type", "tags", *STRING_FIELDS, *INTEGER_FIELDS))
LINK_PATTERN = re.compile(
    r"(?<!!)\[[^\]]*\]\(([^)\s]+?\.md)(?:\s+\"[^\"]*\")?\)"
)


def now_iso() -> str:
    """Returns the current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def is_scalar(value: object) -> bool:
    """Returns whether `value` is an OKF scalar or list of scalars."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_scalar(item) for item in value)
    return False


def require_string(value: object, field: str) -> str:
    """Returns `value` if it is a string.

    Raises:
        ValidationError: Otherwise.
    """
    if isinstance(value, str):
        return value
    raise errors.ValidationError(
        f"frontmatter field {field!r} must be a string, "
        f"got {type(value).__name__}"
    )


def require_integer(value: object, field: str) -> int:
    """Returns `value` if it is an integer (booleans are rejected).

    Raises:
        ValidationError: Otherwise.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise errors.ValidationError(
        f"frontmatter field {field!r} must be an integer, "
        f"got {type(value).__name__}"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Frontmatter:
    """YAML frontmatter of a concept.

    Attributes:
        type: Mandatory concept type, e.g. "ceng/message".
        title: Short human title.
        description: One-line description.
        resource: Source URI or identifier.
        tags: Free-form labels.
        timestamp: ISO 8601 creation/update time.
        priority: Higher numbers are surfaced first.
        expires_at: ISO 8601 expiry, if any.
        helpful_count: Positive feedback counter.
        harmful_count: Negative feedback counter.
        extra: Producer-defined scalar fields.
    """

    type: str
    title: str = ""
    description: str = ""
    resource: str = ""
    tags: tuple[str, ...] = ()
    timestamp: str = ""
    priority: int = 0
    expires_at: str = ""
    helpful_count: int = 0
    harmful_count: int = 0
    extra: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.type:
            raise errors.ValidationError("frontmatter requires a type")
        for key, value in self.extra.items():
            if key in KNOWN_FIELDS or not is_scalar(value):
                raise errors.ValidationError(
                    f"extra field {key!r} must be a new scalar or list of "
                    "scalars"
                )
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_mapping(self) -> dict[str, Any]:
        """Returns the frontmatter as a dict, omitting empty fields."""
        out: dict[str, Any] = {"type": self.type}
        for field in ("title", "description", "resource"):
            if getattr(self, field):
                out[field] = getattr(self, field)
        if self.tags:
            out["tags"] = list(self.tags)
        for field in ("timestamp", "expires_at"):
            if getattr(self, field):
                out[field] = getattr(self, field)
        for field in INTEGER_FIELDS:
            if getattr(self, field):
                out[field] = getattr(self, field)
        out.update(self.extra)
        return out

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Frontmatter:
        """Builds frontmatter from parsed YAML.

        Raises:
            ValidationError: If `type` is missing or a field is mistyped.
        """
        if "type" not in data:
            raise errors.ValidationError("frontmatter is missing 'type'")
        kwargs: dict[str, Any] = {"type": require_string(data["type"], "type")}
        for field in STRING_FIELDS:
            if field in data:
                kwargs[field] = require_string(data[field], field)
        for field in INTEGER_FIELDS:
            if field in data:
                kwargs[field] = require_integer(data[field], field)
        if "tags" in data:
            tags = data["tags"]
            if not isinstance(tags, list):
                raise errors.ValidationError(
                    "frontmatter 'tags' must be a list"
                )
            kwargs["tags"] = tuple(require_string(t, "tags[]") for t in tags)
        kwargs["extra"] = {
            key: value for key, value in data.items() if key not in KNOWN_FIELDS
        }
        return cls(**kwargs)


@dataclasses.dataclass(frozen=True, slots=True)
class Concept:
    """One OKF concept.

    Attributes:
        frontmatter: Structured metadata.
        body: Markdown body.
        path: Bundle-relative POSIX path, or None before placement.
    """

    frontmatter: Frontmatter
    body: str = ""
    path: pathlib.PurePosixPath | None = None

    def at(self, path: str | pathlib.PurePosixPath) -> Concept:
        """Returns a copy placed at bundle-relative `path`."""
        return dataclasses.replace(self, path=pathlib.PurePosixPath(path))

    def render(self) -> str:
        """Renders OKF markdown text (frontmatter, blank line, body)."""
        block = yaml.safe_dump(
            self.frontmatter.to_mapping(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).rstrip("\n")
        body = self.body
        if body and not body.endswith("\n"):
            body += "\n"
        return f"---\n{block}\n---\n\n{body}"

    @classmethod
    def parse(
        cls, text: str, path: pathlib.PurePosixPath | None = None
    ) -> Concept:
        """Parses OKF markdown, normalising CRLF/CR and a leading BOM.

        Raises:
            ValidationError: For missing/unterminated/invalid frontmatter.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.removeprefix("﻿")
        if not text.startswith("---\n"):
            raise errors.ValidationError(
                "concept must begin with '---' and YAML frontmatter"
            )
        remainder = text[4:]
        end = remainder.find("\n---\n")
        if end == -1 and remainder.endswith("\n---"):
            end = len(remainder) - 4
            tail = ""
        elif end == -1:
            raise errors.ValidationError("frontmatter is not terminated")
        else:
            tail = remainder[end + 5 :]
        try:
            loaded = yaml.safe_load(remainder[:end])
        except yaml.YAMLError as exc:
            raise errors.ValidationError(
                f"invalid frontmatter YAML: {exc}"
            ) from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise errors.ValidationError("frontmatter must be a mapping")
        return cls(Frontmatter.from_mapping(loaded), tail.strip("\n"), path)

    def links(self) -> list[str]:
        """Returns bundle-relative `.md` link targets in document order."""
        return [
            target
            for target in LINK_PATTERN.findall(self.body)
            if "://" not in target
        ]
