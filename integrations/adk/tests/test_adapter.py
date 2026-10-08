"""Google ADK adapter against the real framework message classes."""

import asyncio

import pytest
from support import LONG, make_doc, make_runtime


def run(coro):
    return asyncio.run(coro)


def roundtrip(adapter, items):
    flat = [m for item in items for m in adapter.flatten(item)]
    return flat, adapter.rebuild(flat)


class TestADK:
    types = pytest.importorskip("google.genai.types")
    mod = pytest.importorskip("foveate_adk")

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
