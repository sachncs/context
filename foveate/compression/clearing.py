"""Clearing old tool results while keeping the record that they happened."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from foveate import errors
from foveate import messages as messages_lib
from foveate.compression import base, report

if TYPE_CHECKING:
    from foveate import context as context_lib

MARKER = "[tool result"
PREVIEW_CHARS = 80


@base.Compressor.register("clear_tool_results")
@dataclasses.dataclass(frozen=True)
class ClearToolResults(base.Compressor):
    """Replaces the oldest tool results with a short stub until the budget fits.

    A tool result is a message with role `TOOL`, or one whose text starts with
    `[tool result` (how the framework adapters flatten them). The stub keeps
    the tool name and the start of the result and says how many tokens were
    removed, so the model knows the call happened and can repeat it. No model
    call is made.

    Attributes:
        keep: Newest tool results that are never cleared.
    """

    keep: int = 2

    def __post_init__(self) -> None:
        if self.keep < 0:
            raise errors.ConfigError("keep must be >= 0")

    def is_result(self, message: messages_lib.Message) -> bool:
        """Returns whether `message` is a tool result."""
        return (
            message.role is messages_lib.Role.TOOL
            or message.content.startswith(MARKER)
        )

    def stub(self, message: messages_lib.Message, tokens: int) -> str:
        """Builds the replacement text for a cleared result."""
        label = (
            message.name or " ".join(message.content.split())[:PREVIEW_CHARS]
        )
        return f"[tool result cleared: {label} ({tokens} tokens removed)]"

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        tokenizer = context.runtime.tokenizer
        messages = list(context.messages)
        results = [i for i, m in enumerate(messages) if self.is_result(m)]
        clearable = results[: max(0, len(results) - self.keep)]
        result = context
        for index in clearable:
            before = tokenizer.count(messages[index].content)
            stub = self.stub(messages[index], before)
            if tokenizer.count(stub) >= before:
                continue
            messages[index] = messages[index].with_content(stub)
            trace.add(
                report.StepRecord(
                    name=f"m{index} clear",
                    input_tokens=before,
                    output_tokens=tokenizer.count(stub),
                )
            )
            result = dataclasses.replace(context, messages=tuple(messages))
            if result.token_count <= budget.tokens:
                break
        return result
