import threading

import pytest

from foveate import errors
from foveate.internals import filelock
from foveate.stores import FilesystemNotesStore, MemoryNotesStore, NotesStore
from foveate.stores import base as store_base


@pytest.fixture(params=["memory", "filesystem"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryNotesStore()
    return FilesystemNotesStore(tmp_path / "notes")


class TestNotesStoreContract:
    def test_write_read_delete(self, store):
        written = store.write("a/b.md", "hello", tags=("x", "y"))
        read = store.read("a/b.md")
        assert read.content == "hello" and read.tags == ("x", "y")
        assert read.timestamp == written.timestamp != ""
        assert store.delete("a/b.md") and not store.delete("a/b.md")
        with pytest.raises(errors.ValidationError):
            store.read("a/b.md")

    def test_append_keeps_tags_and_creates(self, store):
        store.append("n.md", "one")
        store.write("t.md", "A", tags=("keep",))
        store.append("t.md", "B")
        assert store.read("n.md").content == "one"
        assert store.read("t.md").content == "AB"
        assert store.read("t.md").tags == ("keep",)

    def test_entries_prefix_and_order(self, store):
        store.write("x/1.md", "a")
        store.write("y/2.md", "b")
        store.write("x/3.md", "c")
        assert [str(n.path) for n in store.entries("x/")] == [
            "x/1.md",
            "x/3.md",
        ]
        assert len(store.entries()) == 3

    def test_eviction_oldest_first_and_pins(self, store):
        store.write("old.md", "x" * 100)
        store.write("pin.md", "y" * 100, tags=(store_base.PINNED,))
        store.write("new.md", "z" * 100)
        removed = store.evict_to(250)
        assert [str(p) for p in removed] == ["old.md"]
        removed = store.evict_to(50)  # pinned note survives even over budget
        assert [str(p) for p in removed] == ["new.md"]
        assert store.total_size() == 100
        assert [str(n.path) for n in store.entries()] == ["pin.md"]

    @pytest.mark.parametrize("bad", ["/abs.md", "../x.md", "a.txt", ""])
    def test_bad_paths(self, store, bad):
        with pytest.raises(errors.ValidationError):
            store.write(bad, "x")


def test_filesystem_tags_and_timestamps_persist_across_instances(tmp_path):
    FilesystemNotesStore(tmp_path).write("n.md", "body", tags=("t",))
    again = FilesystemNotesStore(tmp_path).read("n.md")
    assert again.tags == ("t",) and again.timestamp and again.content == "body"


def test_filesystem_concurrent_appends_are_not_lost(tmp_path):
    store = FilesystemNotesStore(tmp_path)

    def work():
        for _ in range(10):
            store.append("log.md", "x")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.read("log.md").content == "x" * 40


def test_filesystem_skips_corrupt_and_unreadable(tmp_path):
    store = FilesystemNotesStore(tmp_path)
    store.write("ok.md", "fine")
    (tmp_path / "broken.md").write_text("not frontmatter")
    assert [str(n.path) for n in store.entries()] == ["ok.md"]
    with pytest.raises(errors.ValidationError):
        store.read("missing.md")


def test_filesystem_config_error(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    with pytest.raises(errors.ConfigError):
        FilesystemNotesStore(blocker / "sub")


def test_registry_has_both_stores():
    assert {"memory", "filesystem"} <= set(NotesStore.registry.names())


class TestFileLock:
    def test_exclusive_and_released(self, tmp_path):
        lock = tmp_path / "l"
        with filelock.file_lock(lock):
            assert lock.exists()
            with pytest.raises(errors.ConfigError):
                with filelock.file_lock(lock, timeout=0.05):
                    pass  # pragma: no cover
        assert not lock.exists()

    def test_stale_lock_is_broken(self, tmp_path):
        lock = tmp_path / "l"
        lock.write_text("")
        import os
        import time

        os.utime(lock, (time.time() - 1000, time.time() - 1000))
        with filelock.file_lock(lock, timeout=1, stale_after=10):
            assert lock.exists()
