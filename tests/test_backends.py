"""Tests for :mod:`ceng.backends`."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ceng.backends import (
    DEFAULT_BACKEND,
    ENV_BACKEND,
    LiteLLMBackend,
    OpenAIBackend,
    VLLMBackend,
    available_backends,
    get_backend,
    reset_backend,
    set_backend,
)


@pytest.fixture(autouse=True)
def clean_globals(monkeypatch):
    """Reset the module-level backend cache and env var around each test."""
    reset_backend()
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    yield
    reset_backend()


def _install_fake_litellm(monkeypatch, captured, return_text):
    """Install a fake ``litellm`` module that records calls.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        captured: Dict that receives the kwargs of every call.
        return_text: Text the fake ``completion`` returns.
    """
    fake = ModuleType("litellm")

    def completion(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=return_text))]
        )

    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _install_fake_vllm(monkeypatch, captured, return_text):
    """Install a fake ``vllm`` module exposing ``LLM`` and ``SamplingParams``."""
    fake = ModuleType("vllm")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.setdefault("sampling_kwargs", []).append(kwargs)

    class FakeLLM:
        def __init__(self, model):
            captured.setdefault("constructed", []).append(model)

        def generate(self, prompts, params):
            captured.setdefault("prompts", []).extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=return_text)])]

    fake.LLM = FakeLLM
    fake.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)


def _install_fake_openai(monkeypatch, captured, return_text):
    fake = ModuleType("openai")

    def factory(**_):
        captured.setdefault("clients", []).append(_)
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: captured.update(create_kw=kw)
                    or SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=return_text)
                            )
                        ]
                    )
                )
            )
        )

    fake.OpenAI = factory
    monkeypatch.setitem(sys.modules, "openai", fake)


# --- registry / selector ---


def test_available_backends_includes_all_three():
    names = available_backends()
    assert names == ["litellm", "openai", "vllm"]


def test_default_backend_is_litellm():
    assert DEFAULT_BACKEND == "litellm"


def test_get_backend_defaults_to_litellm(monkeypatch):
    captured = {}
    _install_fake_litellm(monkeypatch, captured, "ok")
    backend = get_backend()
    assert backend.name == "litellm"
    assert isinstance(backend, LiteLLMBackend)


def test_get_backend_is_cached(monkeypatch):
    _install_fake_litellm(monkeypatch, {}, "ok")
    a = get_backend()
    b = get_backend()
    assert a is b


def test_set_backend_switches_active(monkeypatch):
    captured = {}
    _install_fake_litellm(monkeypatch, captured, "hello")
    backend = set_backend("litellm")
    assert backend.name == "litellm"
    assert get_backend() is backend


def test_set_backend_honours_env_var(monkeypatch):
    captured = {}
    _install_fake_litellm(monkeypatch, captured, "x")
    monkeypatch.setenv(ENV_BACKEND, "litellm")
    backend = set_backend("")
    assert backend.name == "litellm"


def test_set_backend_with_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        set_backend("does-not-exist")


def test_reset_backend_forces_rebuild(monkeypatch):
    captured = {}
    _install_fake_litellm(monkeypatch, captured, "ok")
    a = get_backend()
    reset_backend()
    b = get_backend()
    assert a is not b


# --- LiteLLMBackend ---


def test_litellm_backend_forwards_kwargs(monkeypatch):
    captured = {}
    _install_fake_litellm(monkeypatch, captured, "the answer")
    backend = LiteLLMBackend()
    out = backend.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=10,
    )
    assert out == "the answer"
    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["max_tokens"] == 10


def test_litellm_backend_raises_on_malformed_response(monkeypatch):
    fake = ModuleType("litellm")
    fake.completion = lambda **_ : SimpleNamespace(broken=True)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    backend = LiteLLMBackend()
    with pytest.raises(RuntimeError, match="could not extract"):
        backend.complete(messages=[], model="x")


# --- VLLMBackend ---


def test_vllm_backend_lazily_constructs_engine(monkeypatch):
    captured: dict = {}
    _install_fake_vllm(monkeypatch, captured, "vllm-text")
    backend = VLLMBackend()
    out = backend.complete(
        messages=[{"role": "user", "content": "ping"}],
        model="my/model",
        temperature=0.0,
        max_tokens=5,
    )
    assert out == "vllm-text"
    assert captured["constructed"] == ["my/model"]
    assert captured["sampling_kwargs"][0] == {"temperature": 0.0, "max_tokens": 5}


def test_vllm_backend_reuses_engine_per_model(monkeypatch):
    captured: dict = {}
    _install_fake_vllm(monkeypatch, captured, "ok")
    backend = VLLMBackend()
    backend.complete(messages=[{"role": "user", "content": "a"}], model="m1")
    backend.complete(messages=[{"role": "user", "content": "b"}], model="m1")
    backend.complete(messages=[{"role": "user", "content": "c"}], model="m2")
    assert captured["constructed"] == ["m1", "m2"]


def test_vllm_messages_to_prompt_renders_roles():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    prompt = __import__("ceng.backends", fromlist=["_messages_to_prompt"])._messages_to_prompt(msgs)
    assert prompt.startswith("system: be brief")
    assert "user: hi" in prompt
    assert prompt.endswith("assistant:")


def test_vllm_messages_to_prompt_handles_content_list():
    msgs = [{"role": "user", "content": [{"text": "part1"}, {"text": "part2"}]}]
    prompt = __import__("ceng.backends", fromlist=["_messages_to_prompt"])._messages_to_prompt(msgs)
    assert "part1part2" in prompt


# --- OpenAIBackend ---


def test_openai_backend_calls_chat_completions(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, "openai-text")
    backend = OpenAIBackend()
    out = backend.complete(
        messages=[{"role": "user", "content": "ping"}],
        model="gpt-4o-mini",
        temperature=0.5,
    )
    assert out == "openai-text"
    assert captured["create_kw"]["model"] == "gpt-4o-mini"
    assert captured["create_kw"]["temperature"] == 0.5


# --- integration with selector ---


def test_set_backend_can_switch_to_vllm_and_openai(monkeypatch):
    captured: dict = {}
    _install_fake_vllm(monkeypatch, captured, "v")
    b = set_backend("vllm")
    assert b.name == "vllm"
    text = b.complete(messages=[{"role": "user", "content": "q"}], model="m")
    assert text == "v"

    captured.clear()
    _install_fake_openai(monkeypatch, captured, "o")
    b = set_backend("openai")
    assert b.name == "openai"
    text = b.complete(messages=[{"role": "user", "content": "q"}], model="m")
    assert text == "o"
