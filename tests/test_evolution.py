import json

import pytest

from foveate import errors
from foveate.evolution import (
    AddOp,
    Bullet,
    Checkpoint,
    CuratorOp,
    DeleteOp,
    Evolver,
    EvolverConfig,
    Grader,
    MergeOp,
    Playbook,
    Sample,
    UpdateOp,
    roles,
)
from foveate.messages import Role
from foveate.tokenizers import HeuristicTokenizer
from tests.test_compression import make_runtime

TOK = HeuristicTokenizer()


class ExactGrader(Grader):
    def is_correct(self, predicted, target):
        return predicted == target


def bullet(content, section="others", helpful=0, harmful=0):
    b = Bullet.new(section, content, "t0")
    return Bullet(
        b.id, b.section, b.content, helpful, harmful, b.created_at, b.updated_at
    )


class TestPlaybook:
    def test_ids_are_deterministic_and_unique(self):
        a = Bullet.new("Formulas & Calculations", "x = y")
        assert a.id == Bullet.new("formulas and calculations", " X = Y ").id
        ids = {Bullet.new("others", f"content {i}").id for i in range(5000)}
        assert len(ids) == 5000
        assert a.id.startswith("fac-")

    def test_text_roundtrip(self):
        pb = Playbook().merge(
            [bullet("one", "strategies_and_insights", 2, 1), bullet("two")]
        )
        again = Playbook.parse(pb.render())
        key = lambda b: (b.id, b.section, b.content, b.helpful_count)  # noqa: E731
        assert sorted(map(key, again.bullets)) == sorted(map(key, pb.bullets))

    def test_parse_unknown_section_and_blank(self):
        pb = Playbook.parse(
            "## My New Section\n[a-1] helpful=0 harmful=0 :: hi"
        )
        assert pb.bullets[0].section == "my_new_section"
        assert "my_new_section" in pb.sections
        assert len(Playbook.parse("")) == 0

    def test_json_roundtrip_preserves_timestamps(self):
        pb = Playbook().merge([bullet("one")])
        assert Playbook.from_json(pb.to_json()) == pb
        with pytest.raises(errors.ValidationError):
            Playbook.from_json("{bad")
        with pytest.raises(errors.ValidationError):
            Bullet.from_mapping({"id": "x"})

    def test_duplicate_ids_rejected(self):
        b = bullet("one")
        with pytest.raises(errors.ValidationError):
            Playbook((b, b))

    def test_merge_dedups_by_content_and_prefers_higher_score(self):
        pb = Playbook().merge([bullet("Same Thing", helpful=1)])
        merged = pb.merge(
            [
                bullet("same thing", helpful=5),
                bullet("different"),
                bullet("same thing", helpful=0),
            ]
        )
        assert len(merged) == 2
        assert merged.get(pb.bullets[0].id).helpful_count == 5

    def test_active_trim_and_stats(self):
        pb = Playbook().merge(
            [bullet(f"strategy number {i} " * 5, helpful=i) for i in range(30)]
        )
        assert pb.active(3)[0].helpful_count == 29
        trimmed = pb.trim(TOK, 200)
        assert TOK.count(trimmed.render()) <= 200
        assert 0 < len(trimmed) < len(pb)
        survivors = {b.helpful_count for b in trimmed.bullets}
        assert max(survivors) == 29  # highest-scoring bullets are kept
        assert pb.trim(TOK, 10**6) == pb
        assert len(pb.trim(TOK, 1)) == 0
        assert "total=30" in pb.stats()

    def test_as_message_and_without(self):
        pb = Playbook().merge([bullet("one")])
        m = pb.as_message()
        assert m.role == Role.SYSTEM and "one" in m.content
        assert len(pb.without([pb.bullets[0].id])) == 0


class TestOperations:
    def test_many_adds_in_one_step_never_collide(self):
        ops = [AddOp("others", f"insight {i}") for i in range(50)]
        pb = Playbook()
        for op in ops:
            pb = op.apply(pb, "now").playbook
        assert len(pb) == 50 and len({b.id for b in pb.bullets}) == 50

    def test_add_duplicate_not_counted(self):
        pb = Playbook()
        first = AddOp("others", "same").apply(pb, "n")
        second = AddOp("others", "same").apply(first.playbook, "n")
        assert (first.added, second.added) == (1, 0)

    def test_update_merge_delete(self):
        a, b = bullet("alpha", helpful=3), bullet("beta", helpful=1, harmful=1)
        pb = Playbook().merge([a, b])
        up = UpdateOp(a.id, "alpha 2").apply(pb, "n").playbook
        assert (
            up.get(a.id).content == "alpha 2" and up.get(a.id).updated_at == "n"
        )
        merged = MergeOp(a.id, b.id).apply(pb, "n")
        assert (
            merged.removed == 1 and merged.playbook.get(a.id).harmful_count == 1
        )
        reverse = MergeOp(b.id, a.id).apply(pb, "n").playbook
        assert reverse.get(a.id) is not None and reverse.get(b.id) is None
        deleted = DeleteOp(a.id).apply(pb, "n")
        assert deleted.removed == 1 and deleted.playbook.get(a.id) is None

    def test_invalid_targets_are_noops(self):
        pb = Playbook().merge([bullet("a")])
        for op in (UpdateOp("zz", "x"), MergeOp("zz", "yy"), DeleteOp("zz")):
            assert op.apply(pb, "n").playbook == pb

    def test_parse_dispatch(self):
        assert CuratorOp.parse(
            {"type": "add", "section": "s", "content": "c"}
        ) == AddOp("s", "c")
        assert CuratorOp.parse(
            {"type": "UPDATE", "bullet_id": "i", "content": "c"}
        ) == UpdateOp("i", "c")
        assert CuratorOp.parse(
            {"type": "MERGE", "keep_id": "a", "drop_id": "b"}
        ) == MergeOp("a", "b")
        assert CuratorOp.parse({"type": "DELETE", "id": "x"}) == DeleteOp("x")
        for bad in (
            "text",
            {"type": "NOPE"},
            {"type": "ADD", "content": ""},
            {"type": "UPDATE", "id": "x"},
            {"type": "MERGE", "keep_id": "a", "drop_id": "a"},
            {"type": "DELETE"},
        ):
            assert CuratorOp.parse(bad) is None


class TestExtractJson:
    def test_forms(self):
        assert roles.extract_json('{"a": 1}') == {"a": 1}
        assert roles.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert roles.extract_json('Sure! {"a": {"b": 2}} done') == {
            "a": {"b": 2}
        }

    @pytest.mark.parametrize("bad", ["", "no json", "[1, 2]", "{broken"])
    def test_invalid(self, bad):
        with pytest.raises(errors.ValidationError):
            roles.extract_json(bad)


def ace_responder(answers, insight="use the formula", ops=None, log=None):
    """Routes by prompt type; `answers` maps question text -> final_answer."""

    def respond(request):
        text = request.messages[-1].content
        if "analysis expert" in text:
            question = text.split("**Question:**")[1].split("**Context:**")[0]
            answer = next(
                (a for q, a in answers.items() if q in question), "unknown"
            )
            if log is not None:
                log.append("generate")
            return json.dumps(
                {
                    "reasoning": "r",
                    "bullet_ids": ["BULLET-X"],
                    "final_answer": answer,
                }
            )
        if "diagnose why" in text:
            if log is not None:
                log.append("reflect")
            return json.dumps(
                {
                    "key_insight": insight,
                    "bullet_tags": [{"id": "BULLET-X", "tag": "harmful"}],
                }
            )
        if log is not None:
            log.append("curate")
        return json.dumps(
            {
                "operations": ops
                if ops is not None
                else [{"type": "ADD", "section": "others", "content": insight}]
            }
        )

    return respond


SAMPLES = [Sample("what is 2+2", "4"), Sample("what is 3+3", "6")]


class TestEvolver:
    def run(self, rt, samples=SAMPLES, playbook=None, config=None, **kw):
        return Evolver(rt, config or EvolverConfig()).evolve(
            playbook, samples, ExactGrader(), **kw
        )

    def test_correct_answers_skip_reflector_and_tag_helpful(self):
        log = []
        rt, _ = make_runtime(
            ace_responder({"2+2": "4", "3+3": "6"}, ops=[], log=log)
        )
        used = bullet("used bullet")
        used = Bullet("BULLET-X", used.section, used.content)
        result = self.run(rt, playbook=Playbook((used,)))
        assert "reflect" not in log and log.count("curate") == 2
        assert result.accuracy == 1.0
        assert result.playbook.get("BULLET-X").helpful_count == 2

    def test_wrong_answers_reflect_tag_harmful_and_curate(self):
        log = []
        rt, _ = make_runtime(ace_responder({}, log=log))
        used = Bullet("BULLET-X", "others", "x")
        result = self.run(rt, playbook=Playbook((used,)))
        assert result.accuracy == 0.0
        assert log.count("reflect") == 2
        assert result.playbook.get("BULLET-X").harmful_count == 2
        assert result.steps[0].bullets_added == 1
        assert any(
            b.content == "use the formula" for b in result.playbook.bullets
        )
        assert result.steps[0].usage.total_tokens > 0

    def test_equivalent_answers_graded_by_grader_not_string_equality(self):
        class Numeric(Grader):
            def is_correct(self, predicted, target):
                return float(predicted) == float(target)

        log = []
        rt, _ = make_runtime(
            ace_responder({"2+2": "4.0", "3+3": "6.0"}, ops=[], log=log)
        )
        result = Evolver(rt).evolve(None, SAMPLES, Numeric())
        assert result.accuracy == 1.0 and "reflect" not in log

    def test_curator_frequency_honored(self):
        log = []
        rt, _ = make_runtime(ace_responder({}, log=log))
        samples = [Sample(f"q{i}", "x") for i in range(6)]
        self.run(rt, samples, config=EvolverConfig(curator_frequency=3))
        assert log.count("curate") == 2

    def test_no_ground_truth_reflects_every_sample_ungraded(self):
        log = []
        rt, _ = make_runtime(ace_responder({"2+2": "4", "3+3": "6"}, log=log))
        result = self.run(rt, config=EvolverConfig(use_ground_truth=False))
        assert log.count("reflect") == 2
        assert all(s.correct is None for s in result.steps)
        assert result.accuracy is None

    def test_reflector_rounds_retry_until_insight(self):
        calls = {"reflect": 0}

        def respond(request):
            text = request.messages[-1].content
            if "analysis expert" in text:
                return json.dumps({"final_answer": "bad", "bullet_ids": []})
            if "diagnose why" in text:
                calls["reflect"] += 1
                insight = "" if calls["reflect"] < 3 else "finally"
                return json.dumps({"key_insight": insight})
            return json.dumps({"operations": []})

        rt, _ = make_runtime(respond)
        result = self.run(rt, SAMPLES[:1])
        assert result.steps[0].reflector_rounds == 3

    def test_cache_does_not_leak_between_datasets(self):
        """Regression: v1 keyed reflections only by (iteration, step, round)."""
        insight_by_question = {}

        def respond(request):
            text = request.messages[-1].content
            if "analysis expert" in text:
                return json.dumps({"final_answer": "wrong", "bullet_ids": []})
            if "diagnose why" in text:
                question = (
                    text.split("**Question:**")[1].split("**Model")[0].strip()
                )
                insight_by_question[question] = f"lesson about {question}"
                return json.dumps(
                    {"key_insight": insight_by_question[question]}
                )
            return json.dumps({"operations": []})

        rt, _ = make_runtime(respond)
        seen = []

        def spy(request):
            if "master curator" in request.messages[-1].content:
                seen.append(request.messages[-1].content)
            return respond(request)

        rt, _ = make_runtime(spy)
        self.run(rt, [Sample("first dataset question", "x")])
        self.run(rt, [Sample("second dataset question", "x")])
        assert "lesson about first dataset question" in seen[0]
        assert "lesson about second dataset question" in seen[1]

    def test_backend_error_isolated_and_counted(self):
        rt, _ = make_runtime(
            ace_responder({"3+3": "6"}, ops=[]),
            steps=[errors.PermanentBackendError("down")],
        )
        result = self.run(rt)
        assert (
            result.steps[0].backend_errors == 1
            and result.steps[0].correct is None
        )
        assert result.steps[1].correct is True

    def test_fail_fast_reraises(self):
        rt, _ = make_runtime(steps=[errors.PermanentBackendError("down")])
        with pytest.raises(errors.PermanentBackendError):
            self.run(rt, config=EvolverConfig(fail_fast=True))

    def test_unparseable_reply_counted(self):
        rt, _ = make_runtime(lambda r: "I refuse to output JSON")
        result = self.run(rt)
        assert [s.parse_failures for s in result.steps] == [1, 1]

    def test_playbook_budget_enforced(self):
        big = [
            {
                "type": "ADD",
                "section": "others",
                "content": f"insight {i} " * 40,
            }
            for i in range(10)
        ]
        rt, _ = make_runtime(ace_responder({}, ops=big))
        result = self.run(rt, config=EvolverConfig(playbook_token_budget=300))
        assert TOK.count(result.playbook.render()) <= 300

    def test_checkpoint_resume(self, tmp_path):
        path = tmp_path / "ckpt.json"
        log = []
        rt, _ = make_runtime(ace_responder({}, log=log))
        samples = [Sample(f"q{i}", "x") for i in range(4)]
        first = self.run(rt, samples[:2], checkpoint=path)
        saved = Checkpoint(path).load()
        assert saved is not None and saved[1] == 2
        log.clear()
        resumed = self.run(rt, samples, checkpoint=path)
        assert len(resumed.steps) == 2  # first two steps were skipped
        assert log.count("generate") == 2
        assert len(resumed.playbook) >= len(first.playbook)

    def test_checkpoint_errors(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{nope")
        with pytest.raises(errors.ValidationError):
            Checkpoint(path).load()
        assert Checkpoint(tmp_path / "missing.json").load() is None

    def test_config_validation(self):
        for bad in (
            {"max_reflector_rounds": 0},
            {"curator_frequency": 0},
            {"playbook_token_budget": 0},
        ):
            with pytest.raises(errors.ConfigError):
                EvolverConfig(**bad)
        rt, _ = make_runtime()
        with pytest.raises(errors.ConfigError):
            self.run(rt, epochs=0)

    def test_multiple_epochs(self):
        log = []
        rt, _ = make_runtime(ace_responder({}, ops=[], log=log))
        result = self.run(rt, epochs=2)
        assert [(s.epoch, s.step) for s in result.steps] == [
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
        ]

    def test_grader_default_feedback(self):
        assert "expected '4'" in ExactGrader().feedback("5", "4")
