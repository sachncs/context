"""Google ADK integration (tested with google-adk 1.10 and 2.11).

    from google.adk.agents import LlmAgent
    from foveate.integrations import adk as foveate_adk

    compressor = foveate_adk.model_callback(runtime, budget=4000)
    agent = LlmAgent(..., before_model_callback=compressor)

The callback rewrites `llm_request.contents` just before each model call and
returns None so the call proceeds with the compressed history.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from google.genai import types

from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.documents import document as document_lib
from foveate.integrations import history, tools

Role = messages_lib.Role


class AdkAdapter(history.HistoryAdapter[types.Content]):
    """Maps ADK/Gemini `Content` objects to plain text."""

    def flatten(self, message: types.Content) -> list[messages_lib.Message]:
        role = Role.ASSISTANT if message.role == "model" else Role.USER
        pieces: list[str] = []
        for part in message.parts or []:
            if part.thought:
                continue
            if part.text:
                pieces.append(part.text)
            elif part.function_call is not None:
                call = part.function_call
                pieces.append(f"[tool call {call.name}({call.args})]")
            elif part.function_response is not None:
                result = part.function_response
                pieces.append(f"[tool result {result.name}: {result.response}]")
        text = "\n".join(pieces)
        if not text:
            return []
        if (
            role is Role.USER
            and message.parts
            and all(p.function_response is not None for p in message.parts)
        ):
            role = Role.ASSISTANT
        return [messages_lib.Message(role, text)]

    def rebuild(
        self, messages: Sequence[messages_lib.Message]
    ) -> list[types.Content]:
        rebuilt: list[types.Content] = []
        for message in messages:
            role = "model" if message.role is Role.ASSISTANT else "user"
            part = types.Part(text=message.content)
            if rebuilt and rebuilt[-1].role == role:
                (rebuilt[-1].parts or []).append(part)
            else:
                rebuilt.append(types.Content(role=role, parts=[part]))
        return rebuilt

    def starts_turn(self, message: types.Content) -> bool:
        return message.role == "user" and any(
            part.text for part in message.parts or []
        )


class ModelCallback:
    """ADK `before_model_callback` that compresses `llm_request.contents`."""

    def __init__(
        self, compressor: history.HistoryCompressor[types.Content]
    ) -> None:
        """Wraps a configured compressor."""
        self.compressor = compressor

    async def __call__(self, callback_context: Any, llm_request: Any) -> None:
        llm_request.contents = await self.compressor.compress(
            list(llm_request.contents)
        )
        return None


def model_callback(
    runtime: runtime_lib.Runtime,
    budget: int,
    method: str = history.DEFAULT_METHOD,
    keep_last: int = 4,
    **options: object,
) -> ModelCallback:
    """Builds a `before_model_callback` that keeps contents within `budget`.

    Args:
        runtime: foveate runtime (backend, cache, tokenizer).
        budget: Token ceiling for the request contents.
        method: foveate compression method for the old turns.
        keep_last: Newest contents kept verbatim.
        **options: Strategy options (see `Context.compress`). With a
            composite method such as the default, key them by stage, e.g.
            `ushape={"head": 1}`.
    """
    return ModelCallback(
        history.HistoryCompressor(
            AdkAdapter(),
            runtime,
            budget,
            method=method,
            keep_last=keep_last,
            options=dict(options),
        )
    )


def document_tools(
    documents: Sequence[document_lib.Document], max_tokens: int = 4000
) -> list[Callable[..., str]]:
    """Returns plain tool functions for `LlmAgent(tools=...)`.

    agent = LlmAgent(..., tools=foveate_adk.document_tools([doc]))
    """
    return tools.DocumentTools(documents, max_tokens=max_tokens).functions()
