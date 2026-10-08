"""Versioned prompt templates.

The template fingerprint participates in every cache key, so editing a prompt
can never serve stale cached completions.
"""

from __future__ import annotations

import dataclasses
import string

from foveate import errors
from foveate.internals import hashing


@dataclasses.dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A system prompt plus a user-message format string.

    Attributes:
        name: Stable identifier.
        version: Bump when wording changes meaningfully.
        system: System prompt text.
        user: `str.format`-style template for the user message.
    """

    name: str
    version: str
    system: str
    user: str

    @property
    def fingerprint(self) -> str:
        """Returns a digest covering name, version and full text."""
        return hashing.fingerprint(
            self.name, self.version, self.system, self.user
        )

    def fields(self) -> frozenset[str]:
        """Returns the placeholder names used by the user template."""
        parsed = string.Formatter().parse(self.user)
        return frozenset(item[1] for item in parsed if item[1])

    def render_user(self, **values: str) -> str:
        """Fills the user template.

        Raises:
            ValidationError: If placeholders and `values` do not match.
        """
        expected = self.fields()
        if expected != set(values):
            raise errors.ValidationError(
                f"prompt {self.name!r} expects {sorted(expected)}, "
                f"got {sorted(values)}"
            )
        return self.user.format(**values)
