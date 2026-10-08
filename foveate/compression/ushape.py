"""U-shape compaction: keep the head and tail, compress the middle."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

from foveate import errors
from foveate import messages as messages_lib
from foveate.compression import base, ppa, report

if TYPE_CHECKING:
    from foveate import context as context_lib

SUMMARY_HEADER = "[Summary of earlier conversation]\n"


class MiddleMode(str, enum.Enum):
    """What to do with the middle of the conversation."""

    SUMMARIZE = "summarize"
    DROP = "drop"


@base.Compressor.register("ushape")
@dataclasses.dataclass(frozen=True)
class UShape(base.Compressor):
    """Keeps the first `head` and last `tail` messages, handles the middle.

    The middle is either dropped (with an omission marker) or flattened to a
    transcript and compressed by an inner strategy (PPA by default), whose
    steps are merged into this run's report.

    Attributes:
        head: Leading messages to keep verbatim.
        tail: Trailing messages to keep verbatim.
        middle: Whether to summarise or drop the middle.
        inner: Strategy used to compress the middle transcript.
        summary_role: Role of the message holding the summary or marker.
    """

    head: int = 2
    tail: int = 4
    middle: MiddleMode = MiddleMode.SUMMARIZE
    inner: base.Compressor = dataclasses.field(
        default_factory=ppa.PartitionSummarizeCombine
    )
    summary_role: messages_lib.Role = messages_lib.Role.USER

    def __post_init__(self) -> None:
        if self.head < 0 or self.tail < 0:
            raise errors.ConfigError("head and tail must be >= 0")

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        messages = context.messages
        if len(messages) <= self.head + self.tail:
            return context
        first = messages[: self.head]
        last = messages[len(messages) - self.tail :] if self.tail else ()
        middle = messages[self.head : len(messages) - self.tail]
        tokenizer = context.runtime.tokenizer
        if MiddleMode(self.middle) is MiddleMode.DROP:
            text = f"[{len(middle)} earlier messages omitted]"
        else:
            kept = sum(tokenizer.count(m.content) for m in first + last)
            target = max(1, budget.tokens - kept)
            transcript = "\n".join(
                f"{m.role.value}: {m.content}" for m in middle
            )
            sub = dataclasses.replace(
                context,
                messages=(
                    messages_lib.Message(messages_lib.Role.USER, transcript),
                ),
                report=None,
            )
            compressed = await self.inner.compress(
                sub, report.Budget(target, report.Overflow.TRUNCATE)
            )
            if compressed.report is not None:
                for record in compressed.report.steps:
                    trace.add(record)
            text = SUMMARY_HEADER + compressed.messages[0].content
        summary = messages_lib.Message(self.summary_role, text)
        return dataclasses.replace(context, messages=(*first, summary, *last))
