import asyncio
import json
import time

import pytest

from ceng import errors, messages, observability, runtime as runtime_lib
from ceng.backends import scripted
from ceng.cache import base, sqlite

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
            {"CENG_CACHE_DIR": "", "CENG_MODEL": "m", "CENG_CONCURRENCY": "3"}
        )
        assert rt.model == "m" and rt.concurrency == 3
        assert isinstance(rt.cache, base.NullCache)

    def test_sqlite_cache(self, tmp_path):
        rt = runtime_lib.Runtime.from_env({"CENG_CACHE_DIR": str(tmp_path)})
        assert isinstance(rt.cache, sqlite.SqliteCache)
        rt.close()

    @pytest.mark.parametrize(
        "env",
        [
            {"CENG_TIMEOUT_SECONDS": "abc"},
            {"CENG_RETRY_ATTEMPTS": "1.5"},
            {"CENG_BACKEND": "nope"},
            {"CENG_CONCURRENCY": "0"},
        ],
    )
    def test_strict(self, env):
        with pytest.raises(errors.ConfigError):
            runtime_lib.Runtime.from_env({"CENG_CACHE_DIR": "", **env})

    def test_context_manager(self, backend):
        with runtime_lib.Runtime(backend=backend) as rt:
            assert rt.model
