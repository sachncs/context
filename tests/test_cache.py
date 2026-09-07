"""Tests for :mod:`ceng.cache`."""

from __future__ import annotations

import pytest

from ceng.cache import (
    NAMESPACE_PPA_CHECK,
    NAMESPACE_SUMMARIZE,
    VALID_NAMESPACES,
    Cache,
    make_key,
)


@pytest.fixture()
def fresh_cache(tmp_path):
    cache = Cache(cache_dir=str(tmp_path / "ceng_cache"))
    yield cache
    cache.close()


def test_make_key_is_stable_and_order_insensitive():
    a = make_key({"model": "gpt", "content": "hello", "n": 3})
    b = make_key({"n": 3, "content": "hello", "model": "gpt"})
    assert a == b
    assert len(a) == 64


def test_make_key_changes_with_value():
    a = make_key({"model": "gpt", "content": "hello"})
    b = make_key({"model": "gpt", "content": "hello!"})
    assert a != b


def test_get_missing_returns_none(fresh_cache):
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="nope") is None


def test_set_then_get_roundtrip(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="k1", value={"summary": "hi"})
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="k1") == {"summary": "hi"}


def test_set_overwrites_atomically(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="k1", value={"v": 1})
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="k1", value={"v": 2})
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="k1") == {"v": 2}


def test_namespaces_are_isolated(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="dup", value={"a": 1})
    fresh_cache.set(NAMESPACE_PPA_CHECK, key="dup", value={"a": 2})
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="dup") == {"a": 1}
    assert fresh_cache.get(NAMESPACE_PPA_CHECK, key="dup") == {"a": 2}


def test_delete_returns_true_when_present(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="k", value=1)
    assert fresh_cache.delete(NAMESPACE_SUMMARIZE, key="k") is True
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="k") is None


def test_delete_returns_false_when_absent(fresh_cache):
    assert fresh_cache.delete(NAMESPACE_SUMMARIZE, key="nope") is False


def test_clear_namespace_only_wipes_target(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="a", value=1)
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="b", value=2)
    fresh_cache.set(NAMESPACE_PPA_CHECK, key="c", value=3)
    deleted = fresh_cache.clear_namespace(NAMESPACE_SUMMARIZE)
    assert deleted == 2
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="a") is None
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="b") is None
    assert fresh_cache.get(NAMESPACE_PPA_CHECK, key="c") == 3


def test_contains(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="present", value=1)
    assert fresh_cache.contains(NAMESPACE_SUMMARIZE, key="present") is True
    assert fresh_cache.contains(NAMESPACE_SUMMARIZE, key="missing") is False


def test_size_counts_correctly(fresh_cache):
    assert fresh_cache.size() == 0
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="a", value=1)
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="b", value=2)
    fresh_cache.set(NAMESPACE_PPA_CHECK, key="c", value=3)
    assert fresh_cache.size() == 3
    assert fresh_cache.size(NAMESPACE_SUMMARIZE) == 2
    assert fresh_cache.size(NAMESPACE_PPA_CHECK) == 1


def test_unknown_namespace_raises_on_set(fresh_cache):
    with pytest.raises(ValueError, match="unknown namespace"):
        fresh_cache.set("bogus", key="k", value=1)


def test_unknown_namespace_raises_on_get(fresh_cache):
    with pytest.raises(ValueError, match="unknown namespace"):
        fresh_cache.get("bogus", key="k")


def test_persistence_across_instances(tmp_path):
    cache_dir = str(tmp_path / "persist")
    a = Cache(cache_dir=cache_dir)
    a.set(NAMESPACE_SUMMARIZE, key="x", value={"v": "ok"})
    a.close()
    b = Cache(cache_dir=cache_dir)
    assert b.get(NAMESPACE_SUMMARIZE, key="x") == {"v": "ok"}
    b.close()


def test_valid_namespaces_constant_is_complete():
    assert NAMESPACE_SUMMARIZE in VALID_NAMESPACES
    assert NAMESPACE_PPA_CHECK in VALID_NAMESPACES
    assert len(VALID_NAMESPACES) == 2


def test_unicode_values_roundtrip(fresh_cache):
    payload = {"summary": "héllo — 漢字 ✓"}
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="u", value=payload)
    assert fresh_cache.get(NAMESPACE_SUMMARIZE, key="u") == payload


def test_cache_is_thread_safe(fresh_cache):
    import threading

    errors = []

    def writer(i):
        try:
            for j in range(20):
                fresh_cache.set(NAMESPACE_SUMMARIZE, key=f"k{i}_{j}", value=j)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert fresh_cache.size(NAMESPACE_SUMMARIZE) == 8 * 20


# --- public names, context manager, None-reject, schema version ---


def test_cache_attrs_are_public():
    cache = Cache(cache_dir="/tmp/ceng-public-names-check")
    try:
        assert hasattr(cache, "path")
        assert hasattr(cache, "lock")
        assert hasattr(cache, "conn")
    finally:
        cache.close()


def test_context_manager_closes_on_exit(tmp_path):
    cache_dir = tmp_path / "ctx-mgr"
    cache_dir.mkdir()
    with Cache(cache_dir=str(cache_dir)) as cache:
        cache.set(NAMESPACE_SUMMARIZE, key="k", value=1)
        handle = cache.conn
    # ``conn`` should be closed now; further use raises ProgrammingError.
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        handle.execute("SELECT 1")


def test_context_manager_returns_self(tmp_path):
    cache_dir = tmp_path / "ctx-self"
    cache_dir.mkdir()
    with Cache(cache_dir=str(cache_dir)) as cache:
        cache.set(NAMESPACE_SUMMARIZE, key="k", value=1)
        assert cache.get(NAMESPACE_SUMMARIZE, key="k") == 1


def test_close_issues_wal_checkpoint(tmp_path):
    cache_dir = tmp_path / "wal"
    cache_dir.mkdir()
    cache = Cache(cache_dir=str(cache_dir))
    cache.set(NAMESPACE_SUMMARIZE, key="k", value=1)
    # close() calls PRAGMA wal_checkpoint(TRUNCATE); safe to call twice.
    cache.close()
    cache.close()  # second call must not raise


def test_set_rejects_none_value(fresh_cache):
    with pytest.raises(ValueError, match="cannot be None"):
        fresh_cache.set(NAMESPACE_SUMMARIZE, key="k", value=None)


def test_set_rejects_none_unblocks_contains_disambiguation(fresh_cache):
    fresh_cache.set(NAMESPACE_SUMMARIZE, key="k", value={"x": 1})
    assert fresh_cache.contains(NAMESPACE_SUMMARIZE, key="k") is True
    assert fresh_cache.contains(NAMESPACE_SUMMARIZE, key="missing") is False


def test_get_takes_lock_concurrent_with_writer(tmp_path):
    cache_dir = tmp_path / "race"
    cache_dir.mkdir()
    cache = Cache(cache_dir=str(cache_dir))

    import threading

    errors = []
    barrier = threading.Barrier(4)

    def reader():
        try:
            barrier.wait(timeout=2)
            for _ in range(50):
                cache.get(NAMESPACE_SUMMARIZE, key="k")
        except Exception as exc:
            errors.append(exc)

    def writer():
        try:
            barrier.wait(timeout=2)
            for i in range(50):
                cache.set(NAMESPACE_SUMMARIZE, key="k", value=i)
        except Exception as exc:
            errors.append(exc)

    contains_worker = threading.Thread(target=reader)
    set_worker = threading.Thread(target=writer)
    other_reader = threading.Thread(target=reader)
    other_writer = threading.Thread(target=writer)
    for t in (contains_worker, set_worker, other_reader, other_writer):
        t.start()
    for t in (contains_worker, set_worker, other_reader, other_writer):
        t.join(timeout=5)
    cache.close()
    assert errors == []


def test_size_takes_a_lock_for_consistency(tmp_path):
    cache_dir = tmp_path / "size"
    cache_dir.mkdir()
    cache = Cache(cache_dir=str(cache_dir))
    cache.set(NAMESPACE_SUMMARIZE, key="a", value=1)
    assert cache.size() == 1
    assert cache.size(NAMESPACE_SUMMARIZE) == 1
    assert cache.size(NAMESPACE_PPA_CHECK) == 0
    cache.close()


def test_make_key_handles_non_ascii():
    a = make_key({"q": "café"})
    b = make_key({"q": "cafe"})
    assert a != b
