import asyncio
import sys
import types

import pytest

from ceng import errors, messages
from ceng.backends import base, providers

REQ = base.Request(
    "m", (messages.Message(messages.Role.USER, "hi"),), max_tokens=10
)


def response(text="out"):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )


def run(coro):
    return asyncio.run(coro)


def test_payload():
    p = providers.payload(REQ)
    assert p["max_tokens"] == 10 and p["messages"][0]["role"] == "user"
    assert "max_tokens" not in providers.payload(
        base.Request("m", REQ.messages)
    )


def test_text_and_usage_helpers():
    assert providers.text_from_choices(response("x")) == "x"
    for bad in (
        response(""),
        response(None),
        types.SimpleNamespace(choices=[]),
    ):
        with pytest.raises(errors.ValidationError):
            providers.text_from_choices(bad)
    assert providers.usage_from_response(response()).total_tokens == 7
    assert providers.usage_from_response(object()).total_tokens == 0


def test_require_missing():
    with pytest.raises(errors.ConfigError, match="pip install"):
        providers.require("definitely_not_a_module_xyz", "extra")


def test_litellm(monkeypatch):
    class Boom(Exception):
        status_code = 503

    state = {"fail": False}

    async def acompletion(**kwargs):
        assert kwargs["num_retries"] == 0
        if state["fail"]:
            raise Boom("down")
        return response()

    monkeypatch.setitem(
        sys.modules, "litellm", types.SimpleNamespace(acompletion=acompletion)
    )
    backend = providers.LiteLLMBackend()
    assert run(backend.complete(REQ)).usage.total_tokens == 7
    state["fail"] = True
    with pytest.raises(errors.TransientBackendError):
        run(backend.complete(REQ))


def test_openai(monkeypatch):
    closed = []

    class Completions:
        async def create(self, **kwargs):
            if kwargs["model"] == "bad":
                raise ValueError("bug")
            return response("o")

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

        async def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=Client)
    )
    backend = providers.OpenAIBackend(base_url="http://x")

    async def go():
        out = await backend.complete(REQ)
        with pytest.raises(errors.PermanentBackendError):
            await backend.complete(base.Request("bad", REQ.messages))
        await backend.aclose()
        return out

    assert run(go()).text == "o"
    assert closed == [True]
    run(providers.OpenAIBackend().aclose())


def test_vllm(monkeypatch):
    class Output:
        prompt_token_ids = [1, 2]
        outputs = [types.SimpleNamespace(text="gen", token_ids=[1, 2, 3])]

    class LLM:
        def __init__(self, model, **kwargs):
            self.model = model

        def chat(self, conversations, params):
            if params.max_tokens == 512 and self.model == "empty":
                return [
                    types.SimpleNamespace(
                        prompt_token_ids=[],
                        outputs=[types.SimpleNamespace(text=" ", token_ids=[])],
                    )
                ]
            return [Output()]

    fake = types.SimpleNamespace(
        LLM=LLM, SamplingParams=lambda **kw: types.SimpleNamespace(**kw)
    )
    monkeypatch.setitem(sys.modules, "vllm", fake)
    backend = providers.VLLMBackend()
    out = run(backend.complete(REQ))
    assert (out.text, out.usage.total_tokens) == ("gen", 5)
    with pytest.raises(errors.ValidationError):
        run(backend.complete(base.Request("empty", REQ.messages)))
