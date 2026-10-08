"""Pydantic AI adapter against the real framework message classes."""

import asyncio

import pytest
from support import LONG, make_doc, make_runtime

from foveate import Role


def run(coro):
    return asyncio.run(coro)


def roundtrip(adapter, items):
    flat = [m for item in items for m in adapter.flatten(item)]
    return flat, adapter.rebuild(flat)


class TestPydanticAI:
    pai = pytest.importorskip("pydantic_ai")
    mod = pytest.importorskip("foveate_pydantic_ai")

    def history(self, turns=6):
        m = self.pai.messages
        items = []
        for i in range(turns):
            items.append(
                m.ModelRequest(parts=[m.UserPromptPart(f"q{i} {LONG}")])
            )
            items.append(m.ModelResponse(parts=[m.TextPart(f"a{i} {LONG}")]))
        return items

    def test_flatten_rebuild_and_turn_boundaries(self):
        m = self.pai.messages
        adapter = self.mod.PydanticAIAdapter()
        items = [
            m.ModelRequest(
                parts=[m.SystemPromptPart("sys"), m.UserPromptPart("hi")]
            ),
            m.ModelResponse(
                parts=[m.ToolCallPart("lookup", {"x": 1}), m.TextPart("ok")]
            ),
            m.ModelRequest(parts=[m.ToolReturnPart("lookup", "found", "c1")]),
        ]
        flat, rebuilt = roundtrip(adapter, items)
        assert [x.role for x in flat[:2]] == [Role.SYSTEM, Role.USER]
        assert any("tool call lookup" in x.content for x in flat)
        assert any("tool result lookup" in x.content for x in flat)
        assert rebuilt and all(
            isinstance(r, (m.ModelRequest, m.ModelResponse)) for r in rebuilt
        )
        assert adapter.starts_turn(items[0]) and not adapter.starts_turn(
            items[1]
        )
        assert not adapter.starts_turn(items[2])

    def test_processor_compresses_old_turns(self):
        rt, _ = make_runtime()
        proc = self.mod.history_processor(rt, budget=1000, method="extractive")
        items = self.history()
        out = run(proc(items))
        assert out[-4:] == items[-4:]
        assert proc.compressor.count(out) <= 1000 < proc.compressor.count(items)

    def test_document_tools_run_through_a_real_agent_toolset(self):
        tools = self.mod.document_tools([make_doc(pages=30)])
        assert [t.name for t in tools] == [
            "read_pages",
            "search_document",
            "document_outline",
        ]
        assert "1,577" in tools[0].function("7")
        schema = tools[1].function_schema.json_schema
        assert "query" in schema["properties"] and "query" in schema["required"]
        agent = self.pai.Agent("test", tools=tools)
        assert "search_document" in agent._function_toolset.tools
