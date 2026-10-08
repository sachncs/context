"""The model-backed examples run unchanged against a real provider."""

import json
import runpy
import sys

import pytest

from tests.integration import conftest
from tests.test_examples import EXAMPLES


@pytest.fixture
def example_env(monkeypatch):
    monkeypatch.setenv("FOVEATE_BACKEND", "openai")
    monkeypatch.setenv("FOVEATE_BASE_URL", conftest.BASE_URL)
    monkeypatch.setenv("FOVEATE_MODEL", conftest.MODEL)
    monkeypatch.setenv(
        "FOVEATE_OPTIONS", json.dumps({"reasoning_effort": "low"})
    )
    monkeypatch.setenv("FOVEATE_CACHE_DIR", str(conftest.CACHE_DIR))
    monkeypatch.setenv("OPENAI_API_KEY", conftest.API_KEY or "")
    monkeypatch.syspath_prepend(str(EXAMPLES))


@pytest.mark.parametrize(
    "name",
    ["02_ask_with_citations.py"],
)
def test_example_runs(name, example_env, capsys):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    assert capsys.readouterr().out.strip()
    assert "common" in sys.modules


def test_runtime_from_env_reaches_the_real_provider(example_env):
    import asyncio

    from foveate import Runtime, messages

    with Runtime.from_env() as runtime:
        out = asyncio.run(
            runtime.complete(
                (
                    messages.Message(
                        messages.Role.USER, "Reply with the word: ok"
                    ),
                ),
                source="t",
                namespace="env",
                max_tokens=300,
            )
        )
    assert out.text.strip() and out.usage.total_tokens > 0
