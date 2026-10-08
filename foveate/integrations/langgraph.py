"""LangGraph / LangChain integration (tested with langgraph 1.2).

As a node in any `StateGraph` with a `messages` channel, or as the
`pre_model_hook` of `create_react_agent`:

    from foveate.integrations import langgraph as foveate_lg

    node = foveate_lg.compression_node(runtime, budget=4000)
    agent = create_react_agent(model, tools, pre_model_hook=node)

By default the node returns `llm_input_messages`, so the model sees the
compressed history while the stored state keeps the full transcript. Pass
`persist=True` to replace the stored history instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core import messages as lc
from langchain_core.tools import StructuredTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.documents import document as document_lib
from foveate.integrations import history, tools

Role = messages_lib.Role


def text_of(content: object) -> str:
    """Flattens LangChain message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "\n".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, Mapping))
        )
    return ""


class LangChainAdapter(history.HistoryAdapter[lc.BaseMessage]):
    """Maps LangChain messages to plain text."""

    def flatten(self, message: lc.BaseMessage) -> list[messages_lib.Message]:
        text = text_of(message.content)
        if isinstance(message, lc.SystemMessage):
            role = Role.SYSTEM
        elif isinstance(message, lc.HumanMessage):
            role = Role.USER
        elif isinstance(message, lc.ToolMessage):
            role, text = Role.ASSISTANT, f"[tool result {message.name}: {text}]"
        else:
            role = Role.ASSISTANT
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                text += f"\n[tool call {call['name']}({call['args']})]"
        text = text.strip()
        return [messages_lib.Message(role, text)] if text else []

    def rebuild(
        self, messages: Sequence[messages_lib.Message]
    ) -> list[lc.BaseMessage]:
        kinds = {
            Role.SYSTEM: lc.SystemMessage,
            Role.USER: lc.HumanMessage,
            Role.ASSISTANT: lc.AIMessage,
            Role.TOOL: lc.AIMessage,
        }
        return [kinds[m.role](content=m.content) for m in messages]

    def starts_turn(self, message: lc.BaseMessage) -> bool:
        return isinstance(message, lc.HumanMessage)


class CompressionNode:
    """Graph node / `pre_model_hook` that compresses `state["messages"]`."""

    def __init__(
        self,
        compressor: history.HistoryCompressor[lc.BaseMessage],
        persist: bool = False,
    ) -> None:
        """Wraps a configured compressor.

        Args:
            compressor: Configured history compressor.
            persist: Replace the stored history (True) or only change what
                the model sees (False).
        """
        self.compressor = compressor
        self.persist = persist

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        original = list(state["messages"])
        compressed = await self.compressor.compress(original)
        if compressed == original:
            return {} if self.persist else {"llm_input_messages": original}
        if self.persist:
            return {
                "messages": [
                    lc.RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *compressed,
                ]
            }
        return {"llm_input_messages": compressed}


def compression_node(
    runtime: runtime_lib.Runtime,
    budget: int,
    method: str = history.DEFAULT_METHOD,
    keep_last: int = 4,
    persist: bool = False,
    **options: object,
) -> CompressionNode:
    """Builds a node that keeps the message list within `budget` tokens.

    Args:
        runtime: foveate runtime (backend, cache, tokenizer).
        budget: Token ceiling for the message list.
        method: foveate compression method for the old turns.
        keep_last: Newest messages kept verbatim.
        persist: Replace stored history instead of only the model input.
        **options: Strategy options (see `Context.compress`). With a
            composite method such as the default, key them by stage, e.g.
            `ushape={"head": 1}`.
    """
    return CompressionNode(
        history.HistoryCompressor(
            LangChainAdapter(),
            runtime,
            budget,
            method=method,
            keep_last=keep_last,
            options=dict(options),
        ),
        persist=persist,
    )


def document_tools(
    documents: Sequence[document_lib.Document], max_tokens: int = 4000
) -> list[StructuredTool]:
    """Returns LangChain tools for `create_react_agent(model, tools)`."""
    toolbox = tools.DocumentTools(documents, max_tokens=max_tokens)
    return [StructuredTool.from_function(fn) for fn in toolbox.functions()]
