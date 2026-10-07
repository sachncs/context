"""Chat message model.

`Message` is the typed replacement for the loose `{"role", "content"}` dicts
used by LLM SDKs. Conversion to and from mappings happens only at API edges.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence

from ceng import errors


class Role(str, enum.Enum):
    """Participant role of a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    """A single chat message with plain-text content.

    Attributes:
        role: Who produced the message.
        content: Message text. Multi-part content is flattened on ingest.
        name: Optional participant or tool name.
    """

    role: Role
    content: str
    name: str | None = None

    def with_content(self, content: str) -> Message:
        """Returns a copy of this message with replaced content."""
        return dataclasses.replace(self, content=content)

    def to_mapping(self) -> dict[str, str]:
        """Returns the provider-style mapping for this message."""
        mapping = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            mapping["name"] = self.name
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Message:
        """Builds a message from a provider-style mapping.

        List-style content (`[{"type": "text", "text": ...}]`) is flattened
        to a newline-joined string; non-text parts are dropped.

        Args:
            mapping: Mapping with at least `role` and `content`.

        Returns:
            The parsed message.

        Raises:
            ValidationError: If the role or content is missing or invalid.
        """
        try:
            role = Role(mapping["role"])
        except (KeyError, ValueError) as exc:
            raise errors.ValidationError(
                f"invalid or missing message role: {mapping!r}"
            ) from exc
        if "content" not in mapping:
            raise errors.ValidationError(f"message has no content: {mapping!r}")
        name = mapping.get("name")
        return cls(
            role=role,
            content=flatten_content(mapping["content"]),
            name=name if isinstance(name, str) else None,
        )


def flatten_content(content: object) -> str:
    """Flattens string or multi-part content into plain text.

    Args:
        content: A string, or a sequence of strings / `{"text": ...}` parts.

    Returns:
        The text content, newline-joined for multi-part input.

    Raises:
        ValidationError: If the content is of an unsupported type.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(
                part.get("text"), str
            ):
                parts.append(part["text"])
        return "\n".join(parts)
    raise errors.ValidationError(
        f"unsupported content type: {type(content).__name__}"
    )
