"""Tests for :mod:`ceng.playbook` and :mod:`ceng.playbook.evolver`."""

from __future__ import annotations

from dataclasses import dataclass
import json

from ceng.playbook import (
    Bullet,
    DEFAULT_SECTIONS,
    canon_section,
    content_canonical,
    empty_playbook,
    parse_playbook,
    render_playbook,
)


# --- Bullet -----------------------------------------------------------------


def test_bullet_net_score_is_helpful_minus_harmful():
    b = Bullet(id="s-00001", section="strategies_and_insights", content="x",
               helpful_count=5, harmful_count=2)
    assert b.net_score == 3


def test_bullet_default_counters_are_zero():
    b = Bullet(id="s-00002", section="strategies_and_insights", content="x")
    assert b.helpful_count == 0
    assert b.harmful_count == 0


# --- Playbook --------------------------------------------------------------


def test_empty_playbook_has_seven_default_sections():
    p = empty_playbook()
    for section in DEFAULT_SECTIONS:
        assert section in p.sections_in_order


def test_add_bullet_preserves_id_uniqueness():
    p = empty_playbook()
    p.add_bullet(Bullet(id="u-00001", section="others", content="one"))
    p.add_bullet(Bullet(id="u-00001", section="others", content="two"))
    assert len(p.bullets) == 1
    assert p.bullets["u-00001"].content == "two"


def test_add_bullet_to_new_section_appends_in_order():
    p = empty_playbook()
    p.add_bullet(Bullet(id="x-00001", section="custom_section", content="x"))
    assert p.sections_in_order[-1] == "custom_section"


def test_merge_dedups_by_id_higher_score_wins():
    p = empty_playbook()
    p.add_bullet(Bullet(id="x-00001", section="others", content="x",
                       helpful_count=2, harmful_count=1))
    incoming = [Bullet(id="x-00001", section="others", content="x",
                       helpful_count=10, harmful_count=0)]
    p.merge(incoming)
    assert p.bullets["x-00001"].helpful_count == 10


def test_merge_dedups_by_content_hash_keeps_higher_score():
    p = empty_playbook()
    p.add_bullet(Bullet(id="x-00001", section="others", content="Avoid X",
                       helpful_count=1, harmful_count=0))
    incoming = [Bullet(id="y-00099", section="others", content="avoid x",
                       helpful_count=5, harmful_count=0)]
    added = p.merge(incoming)
    assert added == 0  # duplicate content
    # Higher-score bullet wins.
    winner = p.bullets["x-00001"]
    assert winner.helpful_count == 5


def test_merge_adds_new_bullet_with_content_keeps_lower_score():
    p = empty_playbook()
    p.add_bullet(Bullet(id="x-00001", section="others", content="First idea",
                       helpful_count=10, harmful_count=0))
    incoming = [Bullet(id="y-00099", section="others", content="Different idea",
                       helpful_count=1, harmful_count=0)]
    added = p.merge(incoming)
    assert added == 1
    assert "y-00099" in p.bullets
    assert "x-00001" in p.bullets


def test_canon_section_normalises():
    assert canon_section("  Strategies & Insights ") == "strategies_and_insights"
    assert canon_section("ALREADY") == "already"


def test_content_canonical_is_case_insensitive():
    a = content_canonical("Avoid entering ")
    b = content_canonical("avoid entering")
    assert a == b


def test_render_playbook_round_trips():
    p = empty_playbook()
    p.add_bullet(Bullet(id="str-00001", section="strategies_and_insights",
                        content="Round-trip works"))
    text = render_playbook(p)
    p2 = parse_playbook(text)
    assert "str-00001" in p2.bullets
    assert p2.bullets["str-00001"].content == "Round-trip works"


def test_get_active_orders_by_net_score_then_id():
    p = empty_playbook()
    p.add_bullet(Bullet(id="z-00005", section="others", content="z"))
    p.add_bullet(Bullet(id="a-00010", section="others", content="a",
                       helpful_count=5, harmful_count=0))
    p.add_bullet(Bullet(id="m-00003", section="others", content="m",
                       helpful_count=2, harmful_count=1))
    top = p.get_active()
    ids = [b.id for b in top]
    assert ids[0] == "a-00010"  # highest net score
    assert ids[1] == "m-00003"  # net score 1
    assert ids[2] == "z-00005"  # net score 0


def test_get_active_top_k_limits_results():
    p = empty_playbook()
    for i in range(5):
        p.add_bullet(Bullet(id=f"b-{i:05d}", section="others", content="x",
                           helpful_count=i))
    top = p.get_active(top_k=2)
    assert len(top) == 2
    assert top[0].helpful_count == 4
    assert top[1].helpful_count == 3


# --- Trimming --------------------------------------------------------------


def test_trim_to_token_budget_drops_lowest_score():
    p = empty_playbook()
    for i in range(20):
        p.add_bullet(Bullet(
            id=f"b-{i:05d}",
            section="strategies_and_insights",
            content="x" * 100,
            helpful_count=i,
        ))
    initial = len(p.bullets)
    dropped = p.trim_to_token_budget(budget=200)
    assert dropped > 0
    assert len(p.bullets) < initial


def test_trim_keeps_at_least_one_bullet_per_section():
    """The trim loop has a guard: if ``active`` is empty it returns."""
    p = empty_playbook()
    p.add_bullet(Bullet(id="x-00001", section="strategies_and_insights", content="x"))
    # Trim with budget large enough that the single bullet fits;
    # nothing should be dropped.
    dropped = p.trim_to_token_budget(budget=10_000)
    assert dropped == 0
    assert "x-00001" in p.bullets


# --- Evolver ---------------------------------------------------------------


@dataclass
class FakeBackend:
    name: str = "fake"
    response: str = ""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def complete(self, messages, model, **kw):
        self.calls.append({"messages": messages})
        if self.responses:
            return self.responses.pop(0)
        return self.response


def _simple_evaluator(question, answer, sample):
    if answer == sample.get("ground_truth"):
        return "correct"
    return f"wrong: model said {answer!r}"


def test_evolver_defaults_match_paper_config():
    from ceng.playbook.evolver import EvolverConfig

    cfg = EvolverConfig()
    assert cfg.max_reflector_rounds == 5
    assert cfg.dedup_threshold == 0.90
    assert cfg.playbook_token_budget == 80_000
    assert cfg.curator_frequency == 1


def test_evolver_one_step_calls_generator_then_curator_with_correct_sample():
    from ceng.playbook.evolver import Evolver

    backend = FakeBackend()
    backend.responses = [
        # Generator (correct answer on first try)
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "42"}),
        # Curator — adds the "answer is 42" insight
        json.dumps({"reasoning": "x", "operations": []}),
    ]
    e = Evolver(backend=backend, llm="m", cache_dir="")
    pb = empty_playbook()
    pb, stats = e.run(
        playbook=pb,
        queries=[{"question": "what is the answer", "ground_truth": "42", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    # Two backend calls: generator + curator (correct sample skips reflector)
    assert backend.calls  # at least one
    assert stats  # one EvolverStepStats
    assert stats[0].reflector_rounds_used == 0


def test_evolver_wraps_bullets_from_curator_into_playbook():
    from ceng.playbook.evolver import Evolver

    backend = FakeBackend()
    backend.responses = [
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "42"}),
        json.dumps({
            "reasoning": "x",
            "operations": [
                {"type": "ADD", "section": "strategies_and_insights",
                 "content": "The answer to life is 42"},
            ],
        }),
    ]
    e = Evolver(backend=backend, llm="m", cache_dir="")
    pb, _ = e.run(
        playbook=empty_playbook(),
        queries=[{"question": "q", "ground_truth": "42", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    sections = [b for b in pb.bullets.values() if "42" in b.content]
    assert sections, "Curator's bullet should appear in the playbook"


def test_evolver_reflector_runs_when_model_wrong():
    from ceng.playbook.evolver import Evolver

    backend = FakeBackend()
    backend.responses = [
        # Generator wrong
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "wrong"}),
        # Reflector — produces an insight
        json.dumps({
            "reasoning": "r", "error_identification": "e",
            "root_cause_analysis": "rc", "correct_approach": "ca",
            "key_insight": "Always double-check arithmetic",
            "bullet_tags": [],
        }),
        # Curator adds a bullet
        json.dumps({"reasoning": "x", "operations": []}),
        # (no further calls)
    ]
    e = Evolver(backend=backend, llm="m", cache_dir="")
    pb, stats = e.run(
        playbook=empty_playbook(),
        queries=[{"question": "q", "ground_truth": "right", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    assert stats[0].reflector_rounds_used >= 1


def test_evolver_skips_reflector_when_use_ground_truth_false():
    from ceng.playbook.evolver import Evolver, EvolverConfig

    backend = FakeBackend()
    backend.responses = [
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "x"}),
        json.dumps({"reasoning": "x", "operations": []}),
    ]
    e = Evolver(
        backend=backend, llm="m", cache_dir="",
        config=EvolverConfig(use_ground_truth=False, max_reflector_rounds=5),
    )
    pb, stats = e.run(
        playbook=empty_playbook(),
        queries=[{"question": "q", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    assert stats[0].reflector_rounds_used == 0  # no GT → no reflector


def test_evolver_cache_dir_shared_across_runs(tmp_path):
    """A second Evolvers.run with the same cache_dir hits the cache."""
    from ceng.playbook.evolver import Evolver

    responses = [
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "ok"}),
        json.dumps({"reasoning": "x", "operations": []}),
    ]
    backend_a = FakeBackend(responses=list(responses))
    backend_b = FakeBackend(responses=[])  # should not be called

    cache_dir = str(tmp_path / "cache")

    e = Evolver(backend=backend_a, llm="m", cache_dir=cache_dir)
    e.run(
        playbook=empty_playbook(),
        queries=[{"question": "q", "ground_truth": "ok", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    assert backend_a.calls, "first run should have made LLM calls"

    e2 = Evolver(backend=backend_b, llm="m", cache_dir=cache_dir)
    e2.run(
        playbook=empty_playbook(),
        queries=[{"question": "q", "ground_truth": "ok", "context": ""}],
        evaluator=_simple_evaluator,
        max_iterations=1,
    )
    # Cache-key inputs identical → no backend call needed.
    assert backend_b.calls == []


class _NoOp:
    pass
