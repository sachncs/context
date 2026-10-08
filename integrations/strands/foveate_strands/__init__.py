"""Strands Agents integration (tested with strands-agents 1.x).

    from strands import Agent
    import foveate_strands

    manager = foveate_strands.conversation_manager(runtime, budget=6000)
    agent = Agent(
        model=model,
        conversation_manager=manager,
        tools=foveate_strands.document_tools([document]),
    )

The manager compresses the older part of `agent.messages` before each turn and
when the model reports a context overflow. A tool call is never separated from
its result, and the newest turns stay verbatim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from strands import tool
from strands.agent.conversation_manager import ConversationManager

from foveate import errors
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.agents import history, tools
from foveate.documents import document as document_lib
from foveate.internals import runner

Role = messages_lib.Role
OVERFLOW_TARGET = 0.5


def block_text(block: Any) -> str:
    """Renders one Strands content block as text ("" for unknown blocks)."""
    if not isinstance(block, dict):
        return ""
    if "text" in block:
        return str(block["text"])
    if "toolUse" in block:
        use = block["toolUse"]
        arguments = json.dumps(use.get("input", {}), ensure_ascii=False)
        return f"[tool call {use.get('name', '')}({arguments})]"
    if "toolResult" in block:
        result = block["toolResult"]
        parts = [
            str(part.get("text", part.get("json", "")))
            for part in result.get("content", [])
            if isinstance(part, dict)
        ]
        return "[tool result: " + " ".join(parts) + "]"
    return ""


class StrandsAdapter(history.HistoryAdapter[dict[str, Any]]):
    """Maps Strands message dictionaries to plain text and back."""

    def flatten(self, message: dict[str, Any]) -> list[messages_lib.Message]:
        text = "\n".join(
            t for t in (block_text(b) for b in message.get("content", [])) if t
        )
        if not text:
            return []
        role = Role.USER if message.get("role") == "user" else Role.ASSISTANT
        return [messages_lib.Message(role, text)]

    def rebuild(
        self, messages: Sequence[messages_lib.Message]
    ) -> list[dict[str, Any]]:
        rebuilt: list[dict[str, Any]] = []
        for message in messages:
            role = "assistant" if message.role is Role.ASSISTANT else "user"
            text = message.content
            if message.role is Role.SYSTEM:
                text = f"[system note] {text}"
            if rebuilt and rebuilt[-1]["role"] == role:
                rebuilt[-1]["content"].append({"text": text})
            else:
                rebuilt.append({"role": role, "content": [{"text": text}]})
        return rebuilt

    def starts_turn(self, message: dict[str, Any]) -> bool:
        return message.get("role") == "user" and any(
            "text" in block for block in message.get("content", [])
        )


class FoveateConversationManager(ConversationManager):
    """A Strands conversation manager that compresses old turns."""

    def __init__(
        self, compressor: history.HistoryCompressor[dict[str, Any]]
    ) -> None:
        """Wraps a configured compressor."""
        super().__init__()
        self.compressor = compressor

    def compress(self, agent: Any, goal: float = 1.0) -> None:
        """Compresses `agent.messages` in place to `goal` of the budget."""
        current = self.compressor.count(agent.messages)
        compressor = self.compressor
        if goal < 1.0:
            compressor = history.HistoryCompressor(
                compressor.adapter,
                compressor.runtime,
                max(1, int(current * goal)),
                method=compressor.method,
                keep_last=compressor.keep_last,
                options=compressor.options,
            )
        compressed = runner.run_sync(compressor.compress(agent.messages))
        agent.messages[:] = compressed

    def apply_management(self, agent: Any, **kwargs: Any) -> None:
        """Keeps the history within budget after each turn."""
        self.compress(agent)

    def reduce_context(
        self, agent: Any, e: Exception | None = None, **kwargs: Any
    ) -> None:
        """Shrinks the history to half its size after a context overflow."""
        before = len(agent.messages)
        self.compress(agent, goal=OVERFLOW_TARGET)
        self.removed_message_count += max(0, before - len(agent.messages))


def conversation_manager(
    runtime: runtime_lib.Runtime,
    budget: int,
    method: str = history.DEFAULT_METHOD,
    keep_last: int = 4,
    trigger: float = 1.0,
    target: float = 1.0,
    **options: object,
) -> FoveateConversationManager:
    """Builds a conversation manager that keeps history within `budget`.

    Args:
        runtime: Foveate runtime (backend, cache, tokenizer).
        budget: Token ceiling for the message history.
        method: Foveate compression method for the old turns.
        keep_last: Newest messages kept verbatim.
        trigger: Fraction of `budget` above which compression starts.
        target: Fraction of `budget` to compress down to.
        **options: Strategy options (see `Context.compress`).

    Raises:
        ConfigError: For an invalid budget, trigger or target.
    """
    if budget < 1:
        raise errors.ConfigError("budget must be >= 1")
    return FoveateConversationManager(
        history.HistoryCompressor(
            StrandsAdapter(),
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
) -> list[Any]:
    """Returns Strands tools: `read_pages`, `search_document`, `document_outline`."""
    toolbox = tools.DocumentTools(documents, max_tokens=max_tokens)
    return [tool(fn) for fn in toolbox.functions()]
