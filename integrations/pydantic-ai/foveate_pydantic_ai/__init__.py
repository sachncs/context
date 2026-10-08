"""Pydantic AI integration (tested with pydantic-ai 2.x).

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ProcessHistory
    import foveate_pydantic_ai as foveate_pai

    compressor = foveate_pai.history_processor(runtime, budget=4000)
    agent = Agent(model, capabilities=[ProcessHistory(compressor)])

Older Pydantic AI versions accept the same callable through
`Agent(..., history_processors=[compressor])`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai import Tool
from pydantic_ai import messages as pai

from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.agents import history, tools
from foveate.documents import document as document_lib

Role = messages_lib.Role


def text_of(content: object) -> str:
    """Returns the text parts of a user-prompt content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "\n".join(item for item in content if isinstance(item, str))
    return ""


class PydanticAIAdapter(history.HistoryAdapter[pai.ModelMessage]):
    """Maps Pydantic AI `ModelRequest`/`ModelResponse` to plain text."""

    def flatten(self, message: pai.ModelMessage) -> list[messages_lib.Message]:
        flat: list[messages_lib.Message] = []
        if isinstance(message, pai.ModelRequest):
            for part in message.parts:
                if isinstance(part, pai.SystemPromptPart):
                    flat.append(messages_lib.Message(Role.SYSTEM, part.content))
                elif isinstance(part, pai.UserPromptPart):
                    flat.append(
                        messages_lib.Message(Role.USER, text_of(part.content))
                    )
                elif isinstance(part, pai.ToolReturnPart):
                    flat.append(
                        messages_lib.Message(
                            Role.ASSISTANT,
                            f"[tool result {part.tool_name}: {part.content}]",
                        )
                    )
                elif isinstance(part, pai.RetryPromptPart):
                    flat.append(
                        messages_lib.Message(
                            Role.ASSISTANT, f"[retry: {part.content}]"
                        )
                    )
        else:
            for reply in message.parts:
                if isinstance(reply, pai.TextPart):
                    flat.append(
                        messages_lib.Message(Role.ASSISTANT, reply.content)
                    )
                elif isinstance(reply, pai.ToolCallPart):
                    flat.append(
                        messages_lib.Message(
                            Role.ASSISTANT,
                            f"[tool call {reply.tool_name}({reply.args})]",
                        )
                    )
        return [m for m in flat if m.content]

    def rebuild(
        self, messages: Sequence[messages_lib.Message]
    ) -> list[pai.ModelMessage]:
        rebuilt: list[pai.ModelMessage] = []
        request: list[pai.ModelRequestPart] = []
        response: list[pai.ModelResponsePart] = []
        for message in messages:
            if message.role is Role.ASSISTANT:
                if request:
                    rebuilt.append(pai.ModelRequest(parts=request))
                    request = []
                response.append(pai.TextPart(content=message.content))
                continue
            if response:
                rebuilt.append(pai.ModelResponse(parts=response))
                response = []
            if message.role is Role.SYSTEM:
                request.append(pai.SystemPromptPart(content=message.content))
            else:
                request.append(pai.UserPromptPart(content=message.content))
        if request:
            rebuilt.append(pai.ModelRequest(parts=request))
        if response:
            rebuilt.append(pai.ModelResponse(parts=response))
        return rebuilt

    def starts_turn(self, message: pai.ModelMessage) -> bool:
        return isinstance(message, pai.ModelRequest) and any(
            isinstance(part, pai.UserPromptPart) for part in message.parts
        )


class HistoryProcessor:
    """Async history processor for `ProcessHistory` / `history_processors`."""

    def __init__(
        self, compressor: history.HistoryCompressor[pai.ModelMessage]
    ) -> None:
        """Wraps a configured compressor."""
        self.compressor = compressor

    async def __call__(
        self, messages: list[pai.ModelMessage]
    ) -> list[pai.ModelMessage]:
        return await self.compressor.compress(messages)


def history_processor(
    runtime: runtime_lib.Runtime,
    budget: int,
    method: str = history.DEFAULT_METHOD,
    keep_last: int = 4,
    trigger: float = 1.0,
    target: float = 1.0,
    **options: object,
) -> HistoryProcessor:
    """Builds a history processor that keeps history within `budget` tokens.

    Args:
        runtime: foveate runtime (backend, cache, tokenizer).
        budget: Token ceiling for the whole message history.
        method: foveate compression method for the old turns.
        keep_last: Newest messages kept verbatim.
        trigger: Fraction of `budget` above which compression starts.
        target: Fraction of `budget` to compress down to.
        **options: Strategy options (see `Context.compress`). With a
            composite method such as the default, key them by stage, e.g.
            `ushape={"head": 1}`.
    """
    return HistoryProcessor(
        history.HistoryCompressor(
            PydanticAIAdapter(),
            runtime,
            budget,
            method=method,
            keep_last=keep_last,
            trigger=trigger,
            target=target,
            options=dict(options),
        )
    )


def document_tools(
    documents: Sequence[document_lib.Document], max_tokens: int = 4000
) -> list[Tool[None]]:
    """Returns `read_pages`, `search_document`, `document_outline` tools.

    agent = Agent(model, tools=foveate_pai.document_tools([doc]))
    """
    toolbox = tools.DocumentTools(documents, max_tokens=max_tokens)
    return [Tool(fn) for fn in toolbox.functions()]
