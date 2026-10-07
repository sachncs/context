"""Offload: move old turns into a notes store, leave a retrieval pointer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ceng import errors
from ceng import messages as messages_lib
from ceng.compression import base, report
from ceng.internals import hashing
from ceng.stores import base as store_base

if TYPE_CHECKING:
    from ceng import context as context_lib


@base.Compressor.register("offload")
@dataclasses.dataclass(frozen=True)
class Offload(base.Compressor):
    """Writes the middle of the conversation to a store and points to it.

    Nothing is lost: the dropped turns are saved verbatim as a note, and the
    context keeps a short message naming the note so an agent can fetch it
    back. Notes are content-addressed, so repeating the call is idempotent.

    Attributes:
        store: Where offloaded transcripts are written.
        head: Leading messages kept verbatim.
        tail: Trailing messages kept verbatim.
        prefix: Note path prefix inside the store.
    """

    store: store_base.NotesStore
    head: int = 1
    tail: int = 4
    prefix: str = "offload"

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
        transcript = "\n\n".join(
            f"## {m.role.value}\n{m.content}" for m in middle
        )
        digest = hashing.fingerprint(transcript)[:16]
        path = f"{self.prefix}/{digest}.md"
        self.store.write(path, transcript, tags=("offload",))
        pointer = messages_lib.Message(
            messages_lib.Role.USER,
            f"[{len(middle)} earlier messages offloaded to note {path}]",
        )
        tokenizer = context.runtime.tokenizer
        trace.add(
            report.StepRecord(
                name=f"offloaded {len(middle)} messages to {path}",
                input_tokens=tokenizer.count(transcript),
                output_tokens=tokenizer.count(pointer.content),
            )
        )
        return dataclasses.replace(context, messages=(*first, pointer, *last))
