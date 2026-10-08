import json

import pytest

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.memory import Memory
from foveate.stores import MemoryNotesStore
from tests import faults


def memory(reply=None):
    runtime = None
    if reply is not None:
        runtime = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(default=reply)
        )
    return Memory(MemoryNotesStore(), runtime)


def test_facts_are_deduplicated_by_content():
    m = memory()
    a = m.remember("The user prefers metric units.")
    b = m.remember("the user prefers metric units.  ")
    assert a.path == b.path and len(m.facts()) == 1


def test_session_notes_are_numbered_and_scoped():
    m = memory()
    m.remember("opened the ticket", session="s1")
    m.remember("found the cause", session="s1")
    m.remember("other work", session="s2")
    assert [n.content for n in m.notes("s1")] == [
        "opened the ticket",
        "found the cause",
    ]
    assert len(m.notes()) == 3


def test_recall_ranks_by_relevance_and_scopes_sessions():
    m = memory()
    m.remember("The deployment region is eu-west-1.")
    m.remember("The user prefers dark mode.")
    m.remember("rotating the database password", session="s1")
    m.remember("unrelated secret of another session", session="s2")
    top = m.recall("which region is deployment in", k=2, session="s1")
    assert top[0].content.startswith("The deployment region")
    texts = " ".join(n.content for n in m.recall("secret", k=5, session="s1"))
    assert "another session" not in texts
    assert memory().recall("nothing here") == []


def test_message_respects_the_token_budget():
    m = memory()
    for i in range(6):
        m.remember(f"fact number {i} about deployment " + "detail " * 20)
    message = m.message("deployment", budget=60)
    assert message is not None and message.content.count("- fact") <= 2
    assert m.message("zzzz", budget=60) is None


def test_consolidate_writes_facts_and_removes_notes():
    reply = lambda r: json.dumps({"facts": ["Region is eu-west-1.", " "]})  # noqa: E731
    m = memory(reply)
    m.remember("we looked at the region, it is eu-west-1", session="s1")
    facts = m.consolidate("s1")
    assert facts == ["Region is eu-west-1."]
    assert m.notes("s1") == [] and len(m.facts()) == 1
    assert m.consolidate("s1") == []  # nothing left


def test_consolidate_needs_a_runtime_and_valid_json():
    m = memory()
    m.remember("x", session="s1")
    with pytest.raises(errors.ConfigError):
        m.consolidate("s1")
    bad = memory(lambda r: json.dumps({"nope": 1}))
    bad.remember("x", session="s1")
    with pytest.raises(errors.ValidationError):
        bad.consolidate("s1")


def test_validation_and_forget():
    m = memory()
    with pytest.raises(errors.ValidationError):
        m.remember("   ")
    with pytest.raises(errors.ValidationError):
        m.remember("x", session="../escape")
    note = m.remember("keep me")
    assert m.forget(note.path) and not m.forget(note.path)
