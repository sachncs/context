"""Backend, resilience and runtime behaviour against a real provider."""

import asyncio

import pytest

from ceng import errors, messages, observability
from ceng import runtime as runtime_lib
from ceng.backends import base, resilient
from ceng.cache import base as cache_base
from tests.integration import conftest

USER = messages.Role.USER


def ask(text, **kw):
    return base.Request(
        conftest.MODEL,
        (messages.Message(USER, text),),
        options=(("reasoning_effort", "low"),),
        **kw,
    )


def test_real_completion_reports_text_usage_and_finish_reason():
    backend = conftest.make_backend()
    out = asyncio.run(
        backend.complete(
            ask("Reply with exactly the word: pong", max_tokens=400)
        )
    )
    assert "pong" in out.text.lower()
    assert out.usage.prompt_tokens > 0 and out.usage.completion_tokens > 0
    assert out.finish_reason == "stop" and not out.truncated
    assert out.seconds > 0


def test_invalid_key_is_permanent_and_not_retried():
    backend = conftest.make_backend(api_key="nvapi-this-key-is-invalid")
    with pytest.raises(errors.PermanentBackendError):
        asyncio.run(backend.complete(ask("hi", max_tokens=20)))


def test_unknown_model_is_permanent():
    backend = conftest.make_backend()
    request = base.Request("no-such-vendor/no-such-model", ask("hi").messages)
    with pytest.raises(errors.PermanentBackendError):
        asyncio.run(backend.complete(request))


def test_cache_serves_second_identical_call(unique):
    metrics = observability.MetricsObserver()
    rt = runtime_lib.Runtime(
        backend=conftest.make_backend(),
        model=conftest.MODEL,
        cache=cache_base.MemoryCache(),
        observers=(metrics,),
        options={"reasoning_effort": "low"},
    )
    prompt = (messages.Message(USER, f"Say the token {unique} back to me."),)

    async def go():
        first = await rt.complete(
            prompt, source="t", namespace="n", max_tokens=400
        )
        second = await rt.complete(
            prompt, source="t", namespace="n", max_tokens=400
        )
        return first, second

    first, second = asyncio.run(go())
    assert not first.cached and second.cached and second.text == first.text
    assert metrics.backend_calls == 1 and metrics.cache_hits == 1


def test_reasoning_model_truncation_is_recovered(unique):
    """With default reasoning effort a tiny cap yields empty content; ceng
    must retry with a larger cap instead of failing."""
    metrics = observability.MetricsObserver()
    rt = runtime_lib.Runtime(
        backend=conftest.make_backend(),
        model=conftest.MODEL,
        cache=cache_base.NullCache(),
        observers=(metrics,),
    )
    prompt = (
        messages.Message(USER, f"[{unique}] Name the capital of France."),
    )
    out = asyncio.run(
        rt.complete(prompt, source="t", namespace="n", max_tokens=16)
    )
    assert "paris" in out.text.lower()
    assert metrics.backend_calls >= 2  # the first attempt was truncated


def test_concurrent_distinct_requests(unique):
    rt = runtime_lib.Runtime(
        backend=conftest.make_backend(model_rate=2.0),
        model=conftest.MODEL,
        cache=cache_base.NullCache(),
        options={"reasoning_effort": "low"},
    )

    async def go():
        return await asyncio.gather(
            *(
                rt.complete(
                    (
                        messages.Message(
                            USER, f"[{unique}] What is {n}+{n}? Number only."
                        ),
                    ),
                    source="t",
                    namespace="n",
                    max_tokens=400,
                )
                for n in range(2, 6)
            )
        )

    results = asyncio.run(go())
    assert [
        str(2 * n) in r.text for n, r in zip(range(2, 6), results, strict=True)
    ] == [True] * 4


def test_retry_policy_against_real_timeout():
    backend = resilient.ResilientBackend(
        conftest.make_backend().inner,
        retry=resilient.RetryPolicy(attempts=2, base_delay=0.01),
        timeout=0.001,
    )
    with pytest.raises(errors.BackendTimeoutError):
        asyncio.run(backend.complete(ask("hi", max_tokens=20)))


def test_litellm_backend_reaches_the_real_provider():
    pytest.importorskip("litellm")
    from ceng.backends import LiteLLMBackend

    backend = LiteLLMBackend(
        base_url=conftest.BASE_URL, api_key=conftest.API_KEY
    )
    request = base.Request(
        f"openai/{conftest.MODEL}",
        (messages.Message(USER, "Reply with exactly the word: pong"),),
        max_tokens=400,
        options=(("reasoning_effort", "low"),),
    )
    out = asyncio.run(backend.complete(request))
    assert "pong" in out.text.lower() and out.usage.total_tokens > 0
    bad = LiteLLMBackend(base_url=conftest.BASE_URL, api_key="nvapi-invalid")
    with pytest.raises(errors.PermanentBackendError):
        asyncio.run(bad.complete(request))
