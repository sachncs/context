"""Sliding window: drop the oldest turns."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from foveate import errors
from foveate import messages as messages_lib
from foveate.compression import base, report

if TYPE_CHECKING:
    from foveate import context as context_lib


@base.Compressor.register("window")
@dataclasses.dataclass(frozen=True)
class SlidingWindow(base.Compressor):
    """Drops the oldest non-system messages until the context fits.

    System messages are always kept.

    Attributes:
        min_messages: Newest non-system messages that are never dropped.
    """

    min_messages: int = 1

    def __post_init__(self) -> None:
        if self.min_messages < 0:
            raise errors.ConfigError("min_messages must be >= 0")

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        tokenizer = context.runtime.tokenizer
        messages = list(context.messages)
        droppable = [
            i
            for i, m in enumerate(messages)
            if m.role != messages_lib.Role.SYSTEM
        ]
        protected = set(droppable[len(droppable) - self.min_messages :])
        total = context.token_count
        dropped: set[int] = set()
        for index in droppable:
            if total <= budget.tokens:
                break
            if index in protected:
                break
            dropped.add(index)
            total -= tokenizer.count(messages[index].content)
        kept = tuple(m for i, m in enumerate(messages) if i not in dropped)
        trace.add(
            report.StepRecord(
                name=f"dropped {len(dropped)} messages",
                input_tokens=context.token_count,
                output_tokens=max(total, 0),
            )
        )
        return dataclasses.replace(context, messages=kept)
