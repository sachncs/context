import asyncio
import time

import pytest

from foveate import errors, messages, observability
from foveate import runtime as runtime_lib
from foveate.cache import base, sqlite
from tests import faults as scripted

USER = [messages.Message(messages.Role.USER, "hello")]


def run(coro):
    return asyncio.run(coro)


class TestSqliteCache:
    def test_roundtrip_persist(self, tmp_path):
        with sqlite.SqliteCache(tmp_path) as c:
            c.set("k", "v")
            assert c.get("k") == "v"
            c.delete("k")
            assert c.get("k") is None
            c.set("a", "1")
            c.clear()
            assert c.get("a") is None
        with sqlite.SqliteCache(tmp_path) as c:
            c.set("p", "q")
        with sqlite.SqliteCache(tmp_path) as c:
            assert c.get("p") == "q"

    def test_ttl(self, tmp_path):
        with sqlite.SqliteCache(tmp_path, ttl_seconds=0.05) as c:
            c.set("k", "v")
            time.sleep(0.1)
            assert c.get("k") is None

    def test_eviction(self, tmp_path):
        with sqlite.SqliteCache(tmp_path, max_entries=2) as c:
            for i in range(4):
                c.set(f"k{i}", "v")
                time.sleep(0.002)
            assert c.get("k0") is None and c.get("k3") == "v"

    def test_corrupt_file_quarantined(self, tmp_path):
        (tmp_path / "cache.db").write_bytes(b"this is not sqlite" * 100)
        with sqlite.SqliteCache(tmp_path) as c:
            c.set("k", "v")
            assert c.get("k") == "v"
        assert (tmp_path / "cache.corrupt").exists()

    def test_config_validation(self, tmp_path):
        with pytest.raises(errors.ConfigError):
            sqlite.SqliteCache(tmp_path, max_entries=0)
        with pytest.raises(errors.ConfigError):
            sqlite.SqliteCache(tmp_path, ttl_seconds=0)
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(errors.ConfigError):
            sqlite.SqliteCache(blocker / "sub")

    def test_closed_cache_degrades_to_miss(self, tmp_path):
        c = sqlite.SqliteCache(tmp_path)
        c.set("k", "v")
        c.close()
        assert c.get("k") is None
        c.set("k", "v")
        c.delete("k")
        c.clear()


def test_memory_and_null():
    m = base.MemoryCache()
    m.set("a", "1")
    assert m.get("a") == "1"
    m.delete("a")
    m.set("b", "2")
    m.clear()
    assert m.get("b") is None
    n = base.NullCache()
    n.set("a", "1")
    n.delete("a")
    n.clear()
    assert n.get("a") is None


class TestRuntimeComplete:
    def call(self, rt, namespace="ns", **kw):
        return run(rt.complete(USER, source="t", namespace=namespace, **kw))

    def test_cache_hit_second_time(self, runtime, backend):
        first = self.call(runtime)
        second = self.call(runtime)
        assert not first.cached and second.cached
        assert len(backend.requests) == 1

    def test_namespace_and_params_change_key(self, runtime, backend):
        self.call(runtime, namespace="a")
        self.call(runtime, namespace="b")
        self.call(runtime, namespace="a", temperature=0.7)
        self.call(runtime, namespace="a", max_tokens=9)
        assert len(backend.requests) == 4

    def test_corrupt_cache_value_is_miss(self, runtime, backend):
        self.call(runtime)
        for key in list(runtime.cache.data):
            runtime.cache.data[key] = "{not json"
        assert not self.call(runtime).cached
        assert len(backend.requests) == 2

    def test_single_flight(self, backend):
        backend.delay = 0.05
        rt = runtime_lib.Runtime(backend=backend)

        async def go():
            return await asyncio.gather(
                *(
                    rt.complete(USER, source="t", namespace="n")
                    for _ in range(5)
                )
            )

        results = run(go())
        assert len(backend.requests) == 1
        assert {r.text for r in results} == {"hello"}

    def test_single_flight_propagates_error(self):
        b = scripted.ScriptedBackend(
            [errors.PermanentBackendError("no")], delay=0.02
        )
        rt = runtime_lib.Runtime(backend=b)

        async def go():
            return await asyncio.gather(
                *(
                    rt.complete(USER, source="t", namespace="n")
                    for _ in range(3)
                ),
                return_exceptions=True,
            )

        results = run(go())
        assert all(isinstance(r, errors.PermanentBackendError) for r in results)

    def test_empty_output_regenerated_once(self, backend):
        backend.pending.extend(["  ", "good"])
        rt = runtime_lib.Runtime(backend=backend)
        assert run(rt.complete(USER, source="t", namespace="n")).text == "good"

    def test_empty_output_repeatedly_fails(self, backend):
        backend.pending.extend(["", " ", "x"])
        rt = runtime_lib.Runtime(backend=backend)
        with pytest.raises(errors.ValidationError):
            run(rt.complete(USER, source="t", namespace="n"))

    def test_events_emitted(self, backend):
        metrics = observability.MetricsObserver()
        rt = runtime_lib.Runtime(backend=backend, observers=(metrics,))
        run(rt.complete(USER, source="t", namespace="n"))
        run(rt.complete(USER, source="t", namespace="n"))
        assert (metrics.cache_misses, metrics.cache_hits) == (1, 1)
        assert metrics.backend_calls == 1


class TestFromEnv:
    def test_defaults_disabled_cache(self):
        rt = runtime_lib.Runtime.from_env(
            {
                "FOVEATE_CACHE_DIR": "",
                "FOVEATE_MODEL": "m",
                "FOVEATE_CONCURRENCY": "3",
            }
        )
        assert rt.model == "m" and rt.concurrency == 3
        assert isinstance(rt.cache, base.NullCache)

    def test_rate_limit_and_deadline_from_env(self):
        rt = runtime_lib.Runtime.from_env(
            {
                "FOVEATE_CACHE_DIR": "",
                "FOVEATE_RATE_LIMIT_PER_SECOND": "5",
                "FOVEATE_DEADLINE_SECONDS": "30",
            }
        )
        assert rt.backend.rate_per_second == 5.0 and rt.backend.deadline == 30.0
        with pytest.raises(errors.ConfigError):
            runtime_lib.Runtime.from_env(
                {"FOVEATE_CACHE_DIR": "", "FOVEATE_DEADLINE_SECONDS": "soon"}
            )

    def test_tokenizer_follows_model(self):
        pytest.importorskip("tiktoken")
        rt = runtime_lib.Runtime.from_env(
            {"FOVEATE_CACHE_DIR": "", "FOVEATE_MODEL": "gpt-4o-mini"}
        )
        assert rt.tokenizer.encoding_name == "o200k_base"

    def test_sqlite_cache(self, tmp_path):
        rt = runtime_lib.Runtime.from_env({"FOVEATE_CACHE_DIR": str(tmp_path)})
        assert isinstance(rt.cache, sqlite.SqliteCache)
        rt.close()

    @pytest.mark.parametrize(
        "env",
        [
            {"FOVEATE_TIMEOUT_SECONDS": "abc"},
            {"FOVEATE_RETRY_ATTEMPTS": "1.5"},
            {"FOVEATE_BACKEND": "nope"},
            {"FOVEATE_CONCURRENCY": "0"},
        ],
    )
    def test_strict(self, env):
        with pytest.raises(errors.ConfigError):
            runtime_lib.Runtime.from_env({"FOVEATE_CACHE_DIR": "", **env})

    def test_context_manager(self, backend):
        with runtime_lib.Runtime(backend=backend) as rt:
            assert rt.model


def test_single_flight_follower_survives_cancelled_leader():
    """A cancelled leader must not cancel callers that merely shared it."""
    backend = scripted.ScriptedBackend(delay=0.1)
    rt = runtime_lib.Runtime(backend=backend)

    async def go():
        leader = asyncio.ensure_future(
            rt.complete(USER, source="t", namespace="n")
        )
        await asyncio.sleep(0.02)  # leader is now in flight
        follower = asyncio.ensure_future(
            rt.complete(USER, source="t", namespace="n")
        )
        await asyncio.sleep(0.02)
        leader.cancel()
        result = await follower
        with pytest.raises(asyncio.CancelledError):
            await leader
        return result

    assert run(go()).text == "hello"


def test_single_flight_cancelling_a_follower_leaves_leader_running():
    backend = scripted.ScriptedBackend(delay=0.1)
    rt = runtime_lib.Runtime(backend=backend)

    async def go():
        leader = asyncio.ensure_future(
            rt.complete(USER, source="t", namespace="n")
        )
        await asyncio.sleep(0.02)
        follower = asyncio.ensure_future(
            rt.complete(USER, source="t", namespace="n")
        )
        await asyncio.sleep(0.02)
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        return await leader

    assert run(go()).text == "hello"
    assert len(backend.requests) == 1


class TestTruncationAndOptions:
    def truncated(self):
        from foveate.backends import base as backend_base

        return backend_base.Completion(text="", finish_reason="length")

    def test_empty_truncated_output_is_retried_with_larger_cap(self):
        backend = scripted.ScriptedBackend(
            [self.truncated(), self.truncated(), "answer"]
        )
        rt = runtime_lib.Runtime(backend=backend)
        out = run(rt.complete(USER, source="t", namespace="n", max_tokens=10))
        assert out.text == "answer"
        assert [r.max_tokens for r in backend.requests] == [10, 40, 160]

    def test_truncation_gives_up_after_bounded_retries(self):
        backend = scripted.ScriptedBackend([self.truncated()] * 10)
        rt = runtime_lib.Runtime(backend=backend)
        with pytest.raises(errors.ValidationError, match="token cap"):
            run(rt.complete(USER, source="t", namespace="n", max_tokens=10))
        assert len(backend.requests) == 1 + runtime_lib.LENGTH_RETRIES

    def test_partial_text_on_length_is_accepted(self):
        from foveate.backends import base as backend_base

        backend = scripted.ScriptedBackend(
            [
                backend_base.Completion(
                    text="cut off mid", finish_reason="length"
                )
            ]
        )
        rt = runtime_lib.Runtime(backend=backend)
        out = run(rt.complete(USER, source="t", namespace="n", max_tokens=4))
        assert out.text == "cut off mid" and len(backend.requests) == 1

    def test_options_reach_the_request_and_change_the_cache_key(self):
        backend = scripted.ScriptedBackend()
        low = runtime_lib.Runtime(
            backend=backend, options={"reasoning_effort": "low"}
        )
        high = runtime_lib.Runtime(
            backend=backend,
            cache=low.cache,
            options={"reasoning_effort": "high"},
        )
        run(low.complete(USER, source="t", namespace="n"))
        run(high.complete(USER, source="t", namespace="n"))
        assert len(backend.requests) == 2
        assert backend.requests[0].options == (("reasoning_effort", "low"),)

    def test_without_llm_refuses_to_call_a_model(self):
        rt = runtime_lib.Runtime.without_llm()
        with pytest.raises(
            errors.PermanentBackendError, match="no LLM backend"
        ):
            run(rt.complete(USER, source="t", namespace="n"))

    def test_env_base_url_and_options(self):
        rt = runtime_lib.Runtime.from_env(
            {
                "FOVEATE_CACHE_DIR": "",
                "FOVEATE_BASE_URL": "http://localhost:9/v1",
                "FOVEATE_OPTIONS": '{"reasoning_effort": "low"}',
            }
        )
        assert rt.options == {"reasoning_effort": "low"}
        assert rt.backend.inner.base_url == "http://localhost:9/v1"

    @pytest.mark.parametrize(
        "env",
        [
            {"FOVEATE_OPTIONS": "{bad"},
            {"FOVEATE_OPTIONS": "[1]"},
            {"FOVEATE_BACKEND": "none", "FOVEATE_BASE_URL": "http://x"},
        ],
    )
    def test_env_rejects_bad_values(self, env):
        with pytest.raises(errors.ConfigError):
            runtime_lib.Runtime.from_env({"FOVEATE_CACHE_DIR": "", **env})

    def test_short_truncated_text_means_hidden_reasoning_and_is_retried(self):
        from foveate.backends import base as backend_base

        starved = backend_base.Completion(text="Not", finish_reason="length")
        backend = scripted.ScriptedBackend([starved, "A proper full answer."])
        rt = runtime_lib.Runtime(backend=backend)
        out = run(rt.complete(USER, source="t", namespace="n", max_tokens=100))
        assert out.text == "A proper full answer."
        assert [r.max_tokens for r in backend.requests] == [100, 400]


class TestInlineReasoning:
    @pytest.mark.parametrize(
        "raw,clean",
        [
            ("<think>plan {x}</think>\n\n144", "144"),
            ("<think>still thinking...", ""),
            ("thoughts\n</think>\nanswer", "answer"),
            ("a <b>bold</b> claim", "a <b>bold</b> claim"),
            ("<THINK>x</THINK>ok", "ok"),
        ],
    )
    def test_strip_reasoning(self, raw, clean):
        from foveate.internals import reasoning

        assert reasoning.strip_reasoning(raw) == clean

    def test_runtime_returns_only_the_visible_answer(self):
        from foveate import Message, Role
        from tests.test_compression import make_runtime

        rt, _ = make_runtime(responder=lambda r: '<think>hmm {</think>{"a": 1}')
        out = asyncio.run(
            rt.complete(
                (Message(Role.USER, "q"),),
                source="t",
                namespace="n",
                max_tokens=50,
            )
        )
        assert out.text == '{"a": 1}'


class TestWindowAwareRetry:
    def test_growth_is_capped_to_the_context_window(self):
        from foveate import Message, Role
        from foveate.backends import base

        cut = base.Completion(text="", finish_reason="length")
        backend = scripted.ScriptedBackend([cut] * 6)
        rt = runtime_lib.Runtime(backend=backend, context_window=3000)
        with pytest.raises(errors.ValidationError):
            asyncio.run(
                rt.complete(
                    (Message(Role.USER, "word " * 400),),
                    source="t",
                    namespace="n",
                    max_tokens=1000,
                )
            )
        caps = [r.max_tokens for r in backend.requests]
        assert caps[0] == 1000 and len(caps) >= 2
        assert max(caps) < 3000 and caps == sorted(caps)

    def test_cap_to_window_without_a_known_window_is_unchanged(self):
        from foveate import Message, Role
        from foveate.backends import base

        rt = runtime_lib.Runtime(
            backend=scripted.ScriptedBackend(), model="unknown-model"
        )
        request = base.Request(
            messages=(Message(Role.USER, "hi"),), model="unknown-model"
        )
        assert rt.cap_to_window(request, 5000) == 5000
        small = runtime_lib.Runtime(
            backend=scripted.ScriptedBackend(),
            model="unknown-model",
            context_window=1000,
        )
        assert small.cap_to_window(request, 5000) <= 1000
