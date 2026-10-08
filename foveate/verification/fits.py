"""LLM-free verifier: does the context fit a token limit?"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from foveate import errors
from foveate.verification import base

if TYPE_CHECKING:
    from foveate import context as context_lib


@base.Verifier.register("fits")
@dataclasses.dataclass(frozen=True)
class FitsBudget(base.Verifier):
    """Passes when the context holds at most `tokens` tokens.

    Attributes:
        tokens: Token ceiling.
    """

    tokens: int

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise errors.ConfigError("tokens must be >= 1")

    async def verify(self, context: context_lib.Context) -> base.Verdict:
        actual = context.token_count
        return base.Verdict(
            self.name,
            actual <= self.tokens,
            f"{actual} tokens (limit {self.tokens})",
        )
