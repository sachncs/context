"""LangGraph adapter against the real framework message classes."""

import asyncio

import pytest
from support import LONG, make_doc, make_runtime

from foveate import Message, Role


def run(coro):
    return asyncio.run(coro)


def roundtrip(adapter, items):
    flat = [m for item in items for m in adapter.flatten(item)]
    return flat, adapter.rebuild(flat)


class TestLangGraph:
    lc = pytest.importorskip("langchain_core.messages")
    mod = pytest.importorskip("foveate_langgraph")

    def messages(self, turns=6):
        out = []
        for i in range(turns):
            out.append(self.lc.HumanMessage(f"q{i} {LONG}", id=f"h{i}"))
            out.append(self.lc.AIMessage(f"a{i} {LONG}", id=f"a{i}"))
        return out

    def test_flatten_rebuild_and_turn_boundaries(self):
        lc = self.lc
        adapter = self.mod.LangChainAdapter()
        items = [
            lc.SystemMessage("sys"),
            lc.HumanMessage("hi"),
            lc.AIMessage(
                "", tool_calls=[{"name": "f", "args": {"a": 1}, "id": "c1"}]
            ),
            lc.ToolMessage("done", tool_call_id="c1"),
        ]
        flat, rebuilt = roundtrip(adapter, items)
        assert flat[0] == Message(Role.SYSTEM, "sys")
        assert any("tool result" in x.content for x in flat)
        assert rebuilt
        assert adapter.starts_turn(items[1]) and not adapter.starts_turn(
            items[3]
        )

    def test_node_returns_model_input_or_replaces_history(self):
        rt, _ = make_runtime()
        items = self.messages()
        node = self.mod.compression_node(rt, budget=1000, method="extractive")
        update = run(node({"messages": items}))
        assert set(update) == {"llm_input_messages"}
        assert update["llm_input_messages"][-4:] == items[-4:]
        persist = self.mod.compression_node(
            rt, budget=1000, method="extractive", persist=True
        )
        replaced = run(persist({"messages": items}))["messages"]
        assert replaced[0].__class__.__name__ == "RemoveMessage"
        small = [self.lc.HumanMessage("hi")]
        assert run(persist({"messages": small})) == {}
        assert run(node({"messages": small})) == {"llm_input_messages": small}

    def test_document_tools_invoke_like_langchain_tools(self):
        tools = {
            t.name: t for t in self.mod.document_tools([make_doc(pages=30)])
        }
        assert set(tools) == {
            "read_pages",
            "search_document",
            "document_outline",
        }
        assert "1,577" in tools["read_pages"].invoke({"pages": "7"})
        assert "p.7" in tools["search_document"].invoke(
            {"query": "capital expenditures"}
        )
