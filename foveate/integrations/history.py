"""Framework-neutral compression of an agent's chat history.

Agent frameworks keep rich message objects (tool calls, tool results,
multi-part content). `HistoryCompressor` compresses the *old* part of such a
history and leaves the most recent turns untouched, so a tool call is never
separated from its result. A `HistoryAdapter` teaches it a framework's
message type.

Algorithm:
    1. If the whole history already fits the budget, return it unchanged.
    2. Cut the history at a turn boundary so that the newest `keep_last`
       messages (rounded back to a boundary) stay verbatim.
    3. Flatten the older messages to plain text (tool calls and results are
       rendered as text), compress them with a foveate strategy into the budget
       that the recent turns leave free, and rebuild native messages.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Generic, TypeVar

from foveate import Context, errors
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.compression import report as report_lib

M = TypeVar("M")

DEFAULT_METHOD = "ushape|extractive"


class HistoryAdapter(abc.ABC, Generic[M]):
    """Translates a framework's message type to and from plain text."""

    @abc.abstractmethod
    def flatten(self, message: M) -> list[messages_lib.Message]:
        """Renders one native message as plain-text foveate messages."""

    @abc.abstractmethod
    def rebuild(self, messages: Sequence[messages_lib.Message]) -> list[M]:
        """Builds native messages from compressed plain-text messages."""

    @abc.abstractmethod
    def starts_turn(self, message: M) -> bool:
        """Returns whether history may be cut just before `message`.

        True for a message that opens a user turn; False for tool results
        and other messages that must stay attached to the one before.
        """


@dataclasses.dataclass(frozen=True)
class HistoryCompressor(Generic[M]):
    """Keeps an agent's history inside a token budget.

    Attributes:
        adapter: Knows the framework's message type.
        runtime: Backend, cache and tokenizer used for compression.
        budget: Token ceiling for the whole history.
        method: foveate method for the old part. The default summarises the old
            turns as one transcript (leading system messages stay verbatim)
            and degrades to an offline method if the model is unreachable, so
            an agent run is not aborted by the compression step.
        keep_last: Newest messages that stay verbatim (rounded back to a
            turn boundary).
        min_prefix_tokens: Smallest budget ever given to the old part.
        options: Strategy options forwarded to `Context.acompress`.
        reports: Reports of past compressions, newest last (diagnostics).
    """

    adapter: HistoryAdapter[M]
    runtime: runtime_lib.Runtime
    budget: int
    method: str = DEFAULT_METHOD
    keep_last: int = 4
    min_prefix_tokens: int = 64
    options: dict[str, object] = dataclasses.field(default_factory=dict)
    reports: list[report_lib.CompressionReport] = dataclasses.field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.budget < 1 or self.keep_last < 1 or self.min_prefix_tokens < 1:
            raise errors.ConfigError(
                "budget, keep_last and min_prefix_tokens must be >= 1"
            )

    def count(self, history: Sequence[M]) -> int:
        """Returns the plain-text token count of `history`."""
        tokenizer = self.runtime.tokenizer
        return sum(
            tokenizer.count(flat.content)
            for message in history
            for flat in self.adapter.flatten(message)
        )

    def stage_options(
        self, flat: Sequence[messages_lib.Message]
    ) -> dict[str, object]:
        """Returns strategy options, defaulting `ushape` to transcript mode.

        For a `ushape` stage the leading system messages stay as they are
        (`head`) and everything after them is summarised (`tail=0`). User
        options win over these defaults.
        """
        names = [n.strip() for n in self.method.replace("|", "+").split("+")]
        if "ushape" not in names:
            return dict(self.options)
        leading = 0
        for message in flat:
            if message.role is not messages_lib.Role.SYSTEM:
                break
            leading += 1
        defaults: dict[str, object] = {"head": leading, "tail": 0}
        if len(names) == 1:
            return {**defaults, **self.options}
        given = self.options.get("ushape")
        merged = {**defaults, **(given if isinstance(given, dict) else {})}
        return {**self.options, "ushape": merged}

    def split_point(self, history: Sequence[M]) -> int:
        """Returns the index where the verbatim tail starts (0 = none old)."""
        start = max(0, len(history) - self.keep_last)
        for index in range(start, 0, -1):
            if self.adapter.starts_turn(history[index]):
                return index
        return 0

    async def compress(self, history: Sequence[M]) -> list[M]:
        """Returns `history`, compressed if it exceeds the budget.

        The result is the original list when nothing needed to change.

        Raises:
            CompressionError: Only if the chosen method has no fallback and
                its model call fails.
        """
        items = list(history)
        if self.count(items) <= self.budget:
            return items
        cut = self.split_point(items)
        if cut == 0:
            return items
        head, tail = items[:cut], items[cut:]
        flat = [m for message in head for m in self.adapter.flatten(message)]
        room = max(self.min_prefix_tokens, self.budget - self.count(tail))
        context = Context(tuple(flat), self.runtime)
        compressed = await context.acompress(
            self.method,
            budget=report_lib.Budget(room, report_lib.Overflow.TRUNCATE),
            **self.stage_options(flat),
        )
        if compressed.report is not None:
            self.reports.append(compressed.report)
        return [*self.adapter.rebuild(compressed.messages), *tail]
