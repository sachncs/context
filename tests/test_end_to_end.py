"""End-to-end smoke test for v0.4.0.

Exercises every new surface in one chain:

1. ``compact_messages`` collapses a long fake chat history.
2. ``NotesManager`` writes a small note bundle.
3. ``ppa_compress_to_okf(index_only=True)`` writes only ``index.md``
   + the combined summary — no per-leaf files on disk.
4. ``Evolver.run`` walks the sample once and grows the playbook.
5. ``Playbook.merge`` deduplicates a near-identical bullet.

Stays offline. No LLM calls. Backend is a captured-call FakeBackend
that returns canned responses matching the ACE prompt templates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ceng.compact import compact_messages
from ceng.compress import ppa_compress_to_okf
from ceng.notes import NotesManager
from ceng.okf import (
    CENG_BUNDLE_INDEX,
    CENG_COMBINED_SUMMARY,
    CENG_LEAF_SUMMARY,
    find_concept,
)
from ceng.playbook import (
    Bullet,
    empty_playbook,
)
from ceng.playbook.evolver import Evolver


@dataclass
class FakeBackend:
    """Plays back canned responses in order; raises when exhausted."""
    name: str = "fake"
    responses: list = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = []
        else:
            self.responses = list(self.responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, model, **kw):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeBackend ran out of canned responses")
        return self.responses.pop(0)


def _evaluator(question, answer, sample):
    if answer == sample.get("ground_truth"):
        return "correct"
    return f"wrong: model said {answer!r}"


def test_end_to_end_compact_notes_indexed_compress_evolver(tmp_path):
    # 1. A long chat history gets compacted.
    long_chat = (
        [{"role": "system", "content": "You are a careful assistant."}]
        + [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"message number {i} with some content"}
            for i in range(40)
        ]
    )
    backend_for_compact = FakeBackend(
        responses=["# compact summary\n\nkey points: 1, 2, 3"]
    )
    compacted, prov = compact_messages(
        long_chat,
        backend=backend_for_compact,
        llm="m",
        preserve_first=2,
        preserve_last=4,
        cache_dir=str(tmp_path / "compact_cache"),
    )
    assert prov.preserved_first == 2
    assert prov.preserved_last == 4
    # 41 - 2 - 4 = 35 middle messages folded into the summary
    assert prov.summarised_count == 35
    # Compacted shape: head 2 + summary + tail 4 = 7
    assert len(compacted) == 7
    # Backend called exactly once
    assert len(backend_for_compact.calls) == 1

    # 2. NotesManager writes a few agent notes.
    notes = NotesManager(root=tmp_path / "notes")
    notes.write("team-handbook.md", "# Handbook\nBe kind.\n",
                tags=("pinned",))
    notes.write("standup-2026-07-20.md", "todos for today: ship v0.4\n")
    notes.write("decision-caching.md", "we use sqlite for the LLM cache\n")
    assert sorted(notes.list()) == [
        "decision-caching.md",
        "standup-2026-07-20.md",
        "team-handbook.md",
    ]

    # 3. ppa_compress_to_okf with index_only=True writes only
    #    index.md (+ combined.md for multi-leaf). No per-leaf files
    #    on disk — agents fetch them via ceng.okf on demand.
    # Use a modest-length input + a generous partition_max so the
    # partitioner produces exactly two leaves we can script.
    text = (
        "First leaf worth of text with details about topic A. "
        "More on topic A. Still on topic A."
        " || "
        "Second leaf worth of text with details about topic B. "
        "More on topic B. Still on topic B."
    )
    big_messages = [{"role": "user", "content": text}]
    compress_responses = [
        "summary-of-leaf-a",
        "summary-of-leaf-b",
        "summary-of-leaf-c",
        "summary-of-leaf-d",
        "combined-summary-of-both",
    ]
    backend_for_compress = FakeBackend(responses=list(compress_responses))
    concepts = ppa_compress_to_okf(
        big_messages,
        bundle_dir=str(tmp_path / "okf"),
        bundle_name="my-context",
        budget_tokens=20,
        llm="m",
        cache_dir=str(tmp_path / "compress_cache"),
        backend=backend_for_compress,
        summary_max_tokens=10,
        partition_max_tokens=20,
    )
    bundle_dir = tmp_path / "okf" / "my-context"
    on_disk = sorted(p.name for p in bundle_dir.iterdir())
    # index_only=True: only index.md + combined.md
    assert on_disk == ["combined.md", "index.md"]
    # Per-leaf concepts are NOT materialised at all (not in Python
    # memory, not on disk) under index_only. The returned list is
    # just the index + combined.
    types = [c.frontmatter.type for c in concepts]
    assert types.count(CENG_LEAF_SUMMARY) == 0
    assert types.count(CENG_COMBINED_SUMMARY) == 1
    assert types.count(CENG_BUNDLE_INDEX) == 1

    # The index lists every leaf path so callers know what's available
    # on demand.
    index_concept = find_concept(concepts, "index.md")
    assert index_concept is not None
    # The index body mentions both leaves; agents parse it to decide
    # which to fetch.
    assert "leaf-0" in index_concept.body
    assert "leaf-1" in index_concept.body

    # An agent that wants a leaf now fetches it via ceng.okf.
    # Simulate that fetch: re-run the compression with index_only=False
    # so the leaf files are on disk and readable.
    pass

    # 3b. index_only=False for callers that want the v0.3.0 layout.
    backend_for_compress2 = FakeBackend(responses=list(compress_responses))
    ppa_compress_to_okf(
        big_messages,
        bundle_dir=str(tmp_path / "okf2"),
        bundle_name="full-bundle",
        budget_tokens=20,
        llm="m",
        cache_dir=str(tmp_path / "compress_cache2"),
        backend=backend_for_compress2,
        summary_max_tokens=10,
        partition_max_tokens=20,
        index_only=False,
    )
    full_dir = tmp_path / "okf2" / "full-bundle"
    on_disk_full = sorted(p.name for p in full_dir.iterdir())
    assert "combined.md" in on_disk_full
    assert "index.md" in on_disk_full
    leaf_files = [
        p.name for p in full_dir.iterdir()
        if p.name.startswith("leaf-")
    ]
    assert len(leaf_files) >= 2

    # 4. The Evolver runs one step and the Curator adds a bullet.
    ev_backend = FakeBackend(
        responses=[
            # Generator
            json.dumps({"reasoning": "r",
                          "bullet_ids": [],
                          "final_answer": "wrong"}),
            # Reflector (model was wrong, GT is "right")
            json.dumps({
                "reasoning": "r",
                "error_identification": "answer was wrong",
                "root_cause_analysis": "misread the question",
                "correct_approach": "re-read the question",
                "key_insight": "Always read questions twice before answering",
                "bullet_tags": [],
            }),
            # Curator
            json.dumps({
                "reasoning": "x",
                "operations": [
                    {"type": "ADD",
                     "section": "strategies_and_insights",
                     "content": "Always read questions twice before answering"},
                ],
            }),
        ]
    )
    ev = Evolver(backend=ev_backend, llm="m",
                 cache_dir=str(tmp_path / "evolver_cache"))
    pb, stats = ev.run(
        playbook=empty_playbook(),
        queries=[{"question": "What is 1+1?",
                  "ground_truth": "right",
                  "context": ""}],
        evaluator=_evaluator,
        max_iterations=1,
    )
    # Playbook should now contain the Curator's bullet.
    assert any(
        "read questions twice" in b.content
        for b in pb.bullets.values()
    ), f"Curator's bullet missing from playbook: {list(pb.bullets)}"
    # Reflector ran once (model was wrong)
    assert stats[0].reflector_rounds_used == 1
    assert stats[0].bullets_added == 1
    # Three backend calls: Generator + Reflector + Curator
    assert len(ev_backend.calls) == 3

    # 5. Adding a near-identical bullet only adds once (content-hash dedup).
    before = len(pb.bullets)
    pb.merge([Bullet(id="dup-00001",
                     section="strategies_and_insights",
                     content="always read questions twice before answering")])
    assert len(pb.bullets) == before  # dedup'd, not added
    pb.merge([Bullet(id="new-00001",
                     section="strategies_and_insights",
                     content="a genuinely new strategy")])
    assert len(pb.bullets) == before + 1
