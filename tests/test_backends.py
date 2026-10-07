import asyncio
import random

import pytest

from ceng import errors, messages, observability
from ceng.backends import base, resilient
from tests import faults as scripted

USER = (messages.Message(messages.Role.USER, "hello"),)
REQ = base.Request("m", USER)


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_seconds):
    return None


def make(inner, **kw):
    kw.setdefault("sleep", no_sleep)
    kw.setdefault("rng", random.Random(0))
    return resilient.ResilientBackend(inner, **kw)


def test_scripted_replays_and_records():
    b = scripted.ScriptedBackend(["a", lambda r: "b"], default=lambda r: "z")
    assert [run(b.complete(REQ)).text for _ in range(3)] == ["a", "b", "z"]
    assert len(b.requests) == 3
    assert run(scripted.ScriptedBackend().complete(REQ)).text == "hello"


def test_retries_transient_then_succeeds():
    inner = scripted.ScriptedBackend(
        [errors.TransientBackendError("x"), errors.RateLimitError("y"), "ok"]
    )
    events = observability.MetricsObserver()
    backend = make(inner, observers=(events,))
    assert run(backend.complete(REQ)).text == "ok"
    assert events.retries == 2


def test_permanent_not_retried():
    inner = scripted.ScriptedBackend([errors.PermanentBackendError("no"), "ok"])
    with pytest.raises(errors.PermanentBackendError):
        run(make(inner).complete(REQ))
    assert len(inner.requests) == 1


def test_exhaustion_raises_last():
    inner = scripted.ScriptedBackend([errors.TransientBackendError("t")] * 5)
    backend = make(inner, retry=resilient.RetryPolicy(attempts=3))
    with pytest.raises(errors.TransientBackendError):
        run(backend.complete(REQ))
    assert len(inner.requests) == 3


def test_timeout_is_transient():
    inner = scripted.ScriptedBackend(delay=0.2)
    backend = make(inner, timeout=0.01, retry=resilient.RetryPolicy(attempts=2))
    with pytest.raises(errors.BackendTimeoutError):
        run(backend.complete(REQ))
    assert len(inner.requests) == 0 or len(inner.requests) <= 2


def test_circuit_breaker_opens_and_half_opens():
    now = [0.0]
    breaker = resilient.CircuitBreaker(2, 10.0, clock=lambda: now[0])
    inner = scripted.ScriptedBackend(
        [errors.TransientBackendError("t")] * 2 + ["ok"]
    )
    backend = make(
        inner, breaker=breaker, retry=resilient.RetryPolicy(attempts=1)
    )
    for _ in range(2):
        with pytest.raises(errors.TransientBackendError):
            run(backend.complete(REQ))
    with pytest.raises(errors.CircuitOpenError):
        run(backend.complete(REQ))
    now[0] = 11.0
    assert run(backend.complete(REQ)).text == "ok"
    assert breaker.failures == 0


def test_policy_validation_and_delay():
    with pytest.raises(errors.ConfigError):
        resilient.RetryPolicy(attempts=0)
    with pytest.raises(errors.ConfigError):
        resilient.RetryPolicy(base_delay=-1)
    with pytest.raises(errors.ConfigError):
        resilient.CircuitBreaker(0)
    with pytest.raises(errors.ConfigError):
        make(scripted.ScriptedBackend(), timeout=0)
    p = resilient.RetryPolicy(base_delay=1, max_delay=4)
    rng = random.Random(1)
    assert 0 <= p.delay(1, rng, None) <= 1
    assert p.delay(1, rng, 3.0) >= 3.0
    assert p.delay(1, rng, 99.0) == 4.0


def test_concurrency_cap():
    active = {"now": 0, "max": 0}

    async def slow(request):
        return None

    class Probe(base.Backend):
        async def complete(self, request):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return base.Completion("x")

    backend = make(Probe(), max_concurrency=2)

    async def go():
        await asyncio.gather(*(backend.complete(REQ) for _ in range(8)))

    run(go())
    assert active["max"] == 2
    run(backend.aclose())


def test_request_fingerprint_sensitivity():
    a = base.Request("m", USER, temperature=0.0)
    assert a.fingerprint == base.Request("m", USER, temperature=0.0).fingerprint
    assert a.fingerprint != base.Request("m", USER, temperature=0.5).fingerprint
    assert a.fingerprint != base.Request("n", USER).fingerprint
    assert a.fingerprint != base.Request("m", USER, max_tokens=5).fingerprint


class TopJitter(random.Random):
    """Deterministic jitter: always the ceiling of the backoff window."""

    def uniform(self, a, b):
        return b


def test_rate_limit_spaces_call_starts():
    waits = []

    async def record(seconds):
        waits.append(seconds)

    inner = scripted.ScriptedBackend()
    backend = make(inner, rate_per_second=2.0, sleep=record, clock=lambda: 0.0)

    async def go():
        for _ in range(3):
            await backend.complete(REQ)

    run(go())
    assert waits == [0.5, 1.0]
    assert len(inner.requests) == 3


def test_deadline_stops_retrying():
    now = [0.0]

    async def advance(seconds):
        now[0] += seconds

    inner = scripted.ScriptedBackend([errors.TransientBackendError("t")] * 10)
    backend = make(
        inner,
        retry=resilient.RetryPolicy(attempts=10, base_delay=1.0),
        deadline=2.5,
        sleep=advance,
        clock=lambda: now[0],
        rng=TopJitter(),
        breaker=resilient.CircuitBreaker(100),
    )
    with pytest.raises(errors.TransientBackendError):
        run(backend.complete(REQ))
    # attempt 1 -> wait 1s; attempt 2 -> a 2s wait would reach 3s > 2.5s: stop
    assert len(inner.requests) == 2 and now[0] == 1.0


def test_rate_and_deadline_validation():
    with pytest.raises(errors.ConfigError):
        make(scripted.ScriptedBackend(), rate_per_second=0)
    with pytest.raises(errors.ConfigError):
        make(scripted.ScriptedBackend(), deadline=0)
