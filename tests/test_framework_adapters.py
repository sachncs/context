"""Framework adapters against the real message classes (no model, no key)."""

import asyncio

import pytest

from foveate import Message, Role
from tests.test_compression import make_runtime
from tests.test_longdoc import make_doc

LONG = "word " * 120


def run(coro):
    return asyncio.run(coro)


def roundtrip(adapter, items):
    flat = [m for item in items for m in adapter.flatten(item)]
    return flat, adapter.rebuild(flat)


class TestPydanticAI:
    pai = pytest.importorskip("pydantic_ai")
    mod = pytest.importorskip("foveate.integrations.pydantic_ai")

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


class TestADK:
    types = pytest.importorskip("google.genai.types")
    mod = pytest.importorskip("foveate.integrations.adk")

    def contents(self, turns=6):
        t = self.types
        out = []
        for i in range(turns):
            out.append(
                t.Content(role="user", parts=[t.Part(text=f"q{i} {LONG}")])
            )
            out.append(
                t.Content(role="model", parts=[t.Part(text=f"a{i} {LONG}")])
            )
        return out

    def test_flatten_rebuild_and_turn_boundaries(self):
        t = self.types
        adapter = self.mod.AdkAdapter()
        items = [
            t.Content(role="user", parts=[t.Part(text="hi")]),
            t.Content(
                role="model",
                parts=[
                    t.Part(
                        function_call=t.FunctionCall(name="f", args={"a": 1})
                    ),
                    t.Part(text="thinking", thought=True),
                ],
            ),
            t.Content(
                role="user",
                parts=[
                    t.Part(
                        function_response=t.FunctionResponse(
                            name="f", response={"r": 2}
                        )
                    )
                ],
            ),
        ]
        flat, rebuilt = roundtrip(adapter, items)
        assert not any("thinking" in x.content for x in flat)
        assert any("tool call f" in x.content for x in flat)
        assert rebuilt and all(isinstance(r, t.Content) for r in rebuilt)
        assert adapter.starts_turn(items[0]) and not adapter.starts_turn(
            items[2]
        )

    def test_callback_rewrites_request_contents(self):
        rt, _ = make_runtime()
        callback = self.mod.model_callback(rt, budget=1000, method="extractive")

        class Request:
            contents = self.contents()

        request = Request()
        original = list(request.contents)
        assert run(callback(None, request)) is None
        assert request.contents[-4:] == original[-4:]
        assert len(request.contents) < len(original) or (
            callback.compressor.count(request.contents) <= 1000
        )

    def test_document_tools_are_adk_function_tools(self):
        from google.adk.tools import FunctionTool

        fns = self.mod.document_tools([make_doc(pages=30)])
        declarations = [FunctionTool(fn)._get_declaration() for fn in fns]
        assert [d.name for d in declarations] == [
            "read_pages",
            "search_document",
            "document_outline",
        ]
        assert "query" in declarations[1].parameters_json_schema["properties"]


class TestLangGraph:
    lc = pytest.importorskip("langchain_core.messages")
    mod = pytest.importorskip("foveate.integrations.langgraph")

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
