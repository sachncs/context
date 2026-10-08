"""Strands adapter against a real Strands Agent (no model call is made)."""

import asyncio

import foveate_strands
import pytest
from support import LONG, make_doc, make_runtime

from foveate import Role, errors
from foveate.agents import tools


def run(coro):
    return asyncio.run(coro)


def conversation(turns=6):
    items = []
    for i in range(turns):
        items.append({"role": "user", "content": [{"text": f"q{i} {LONG}"}]})
        items.append(
            {"role": "assistant", "content": [{"text": f"a{i} {LONG}"}]}
        )
    return items


class FakeAgent:
    def __init__(self, messages):
        self.messages = messages


def test_flatten_rebuild_and_turn_boundaries():
    adapter = foveate_strands.StrandsAdapter()
    items = [
        {"role": "user", "content": [{"text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "lookup",
                        "input": {"x": 1},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "content": [{"text": "found"}],
                        "status": "success",
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"image": {}}]},
    ]
    flat = [m for item in items for m in adapter.flatten(item)]
    assert [m.role for m in flat] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert "tool call lookup" in flat[1].content and "found" in flat[2].content
    rebuilt = adapter.rebuild(flat)
    assert [r["role"] for r in rebuilt] == ["user", "assistant", "user"]
    assert adapter.starts_turn(items[0]) and not adapter.starts_turn(items[2])


def test_manager_compresses_in_place_and_keeps_the_newest_turns():
    runtime, _ = make_runtime()
    manager = foveate_strands.conversation_manager(
        runtime, budget=1000, method="extractive"
    )
    agent = FakeAgent(conversation())
    newest = agent.messages[-4:]
    before = manager.compressor.count(agent.messages)
    manager.apply_management(agent)
    assert agent.messages[-4:] == newest
    assert manager.compressor.count(agent.messages) <= 1000 < before


def test_reduce_context_halves_the_history_after_an_overflow():
    runtime, _ = make_runtime()
    manager = foveate_strands.conversation_manager(
        runtime, budget=100_000, method="extractive"
    )
    agent = FakeAgent(conversation())
    before = manager.compressor.count(agent.messages)
    manager.reduce_context(agent, RuntimeError("context window exceeded"))
    assert manager.compressor.count(agent.messages) < before * 0.75
    assert manager.removed_message_count >= 0


def test_manager_works_with_a_real_strands_agent():
    from strands import Agent

    runtime, _ = make_runtime()
    manager = foveate_strands.conversation_manager(
        runtime, budget=2000, method="extractive"
    )
    agent = Agent(
        conversation_manager=manager,
        tools=foveate_strands.document_tools([make_doc()]),
        callback_handler=None,
    )
    assert agent.conversation_manager is manager
    assert {"read_pages", "search_document", "document_outline"} <= set(
        agent.tool_names
    )


def test_document_tools_are_strands_tools_that_run():
    found = {
        t.tool_name: t for t in foveate_strands.document_tools([make_doc()])
    }
    assert set(found) == {"read_pages", "search_document", "document_outline"}
    spec = found["search_document"].tool_spec
    assert "query" in spec["inputSchema"]["json"]["properties"]
    assert "1,577" in found["read_pages"]("7")
    assert tools.DocumentTools([make_doc()]).read("7")
    with pytest.raises(errors.ConfigError):
        foveate_strands.conversation_manager(make_runtime()[0], budget=0)
