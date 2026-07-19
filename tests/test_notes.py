"""Tests for :mod:`ceng.notes`."""

from __future__ import annotations

import pytest

from ceng.notes import NotesManager, now_iso


@pytest.fixture()
def root(tmp_path):
    p = tmp_path / "notes"
    p.mkdir()
    return p


def test_write_then_read_round_trips_content(root):
    n = NotesManager(root)
    n.write("a/b.md", "# hello\nworld\n")
    rd = n.read("a/b.md")
    assert rd.content == "# hello\nworld\n"
    assert rd.path == "a/b.md"
    assert rd.timestamp != ""


def test_append_concatenates(root):
    n = NotesManager(root)
    n.write("a/b.md", "first\n")
    n.append("a/b.md", "second\n")
    assert n.read("a/b.md").content == "first\nsecond\n"


def test_append_on_missing_note_creates_it(root):
    n = NotesManager(root)
    n.append("a/brand_new.md", "first\n")
    assert "a/brand_new.md" in n.list()


def test_list_orders_by_mtime(root):
    import time

    n = NotesManager(root)
    n.write("a.md", "first\n")
    time.sleep(0.01)
    n.write("b.md", "second\n")
    time.sleep(0.01)
    n.write("c.md", "third\n")
    assert n.list() == ["a.md", "b.md", "c.md"]


def test_delete_returns_true_then_false(root):
    n = NotesManager(root)
    n.write("a.md", "x")
    assert n.delete("a.md") is True
    assert n.delete("a.md") is False


def test_read_missing_raises(root):
    n = NotesManager(root)
    with pytest.raises(FileNotFoundError):
        n.read("nope.md")


def test_write_rejects_path_traversal(root):
    n = NotesManager(root)
    with pytest.raises(ValueError):
        n.write("../escape.md", "x")


def test_write_rejects_path_traversal_or_dotdot(root):
    n = NotesManager(root)
    with pytest.raises(ValueError):
        n.write("../escape.md", "x")
    with pytest.raises(ValueError):
        n.write("a/../../escape.md", "x")


def test_write_rejects_unsafe_chars(root):
    n = NotesManager(root)
    with pytest.raises(ValueError, match="unsafe"):
        n.write("a$bad.md", "x")


def test_compact_by_size_drops_oldest_first(root):
    import time

    n = NotesManager(root)
    for name, body in [
        ("01.md", "a" * 100),
        ("02.md", "b" * 100),
        ("03.md", "c" * 100),
    ]:
        n.write(name, body)
        time.sleep(0.01)
    # All three together = 300 bytes; budget 250 drops the oldest.
    dropped = n.compact_by_size(max_bytes=250, keep_recent=2)
    remaining = n.list()
    assert dropped == 1
    assert "01.md" not in remaining
    assert "02.md" in remaining
    assert "03.md" in remaining


def test_compact_keeps_keep_recent(root):
    import time

    n = NotesManager(root)
    for name in ["a.md", "b.md", "c.md", "d.md", "e.md"]:
        n.write(name, "x")
        time.sleep(0.01)
    dropped = n.compact_by_size(max_bytes=1, keep_recent=3)
    remaining = n.list()
    # keep_recent=3 wins over size budget.
    assert set(remaining) == {"c.md", "d.md", "e.md"}
    assert dropped == 2


def test_pinned_notes_never_evicted_by_size(root):
    import time

    n = NotesManager(root)
    n.write("_pinned.md", "important")
    for name in ["a.md", "b.md"]:
        n.write(name, "filler")
        time.sleep(0.01)
    n.compact_by_size(max_bytes=1, keep_recent=0)
    # Pinned stays even if size budget can't accommodate it.
    assert "_pinned.md" in n.list()


def test_compact_idempotent(root):
    n = NotesManager(root)
    for name in ["a.md", "b.md", "c.md"]:
        n.write(name, "x")
    first = n.compact_by_size(max_bytes=1, keep_recent=2)
    second = n.compact_by_size(max_bytes=1, keep_recent=2)
    # First pass does the work; second pass has nothing left to drop.
    assert first >= 1
    assert second == 0


def test_write_creates_parent_directories(root):
    n = NotesManager(root)
    n.write("deeply/nested/path/note.md", "x")
    assert (root / "deeply/nested/path/note.md").read_text(encoding="utf-8") == "x"


def test_now_iso_is_iso_format():
    out = now_iso()
    assert isinstance(out, str)
    assert "T" in out
