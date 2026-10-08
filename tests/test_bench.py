import asyncio
import json

import pytest

from foveate import errors
from foveate.bench import Arm, Benchmark, DDXPlus, Finer, Formula, Runner
from tests.test_compression import make_runtime


class TestFixturesAndSeeds:
    @pytest.mark.parametrize("name", ["finer", "formula", "ddxplus"])
    def test_packaged_data_loads(self, name):
        benchmark = Benchmark.registry.get(name)()
        samples = benchmark.load_samples()
        assert len(samples) == 20 and all(s.target for s in samples)
        assert len(benchmark.load_samples(limit=5)) == 5
        assert len(benchmark.seed_playbook()) > 5

    def test_limit_validation_and_registry(self):
        with pytest.raises(errors.ConfigError):
            Finer().load_samples(limit=0)
        assert Benchmark.registry.names() == ["ddxplus", "finer", "formula"]
        assert Finer.name == "finer"

    def test_custom_path_and_bad_data(self, tmp_path):
        good = tmp_path / "g.jsonl"
        good.write_text(
            json.dumps({"question": "q", "options": ["a", "b"], "answer": 0})
            + "\n\n"
        )
        assert DDXPlus().load_samples(path=good)[0].target == "0"  # int 0 kept
        for content in ("{bad", "[1]", json.dumps({"options": ["a"]})):
            bad = tmp_path / "b.jsonl"
            bad.write_text(content)
            with pytest.raises(errors.ValidationError):
                DDXPlus().load_samples(path=bad)
        with pytest.raises(errors.ValidationError):
            DDXPlus().load_samples(path=tmp_path / "missing.jsonl")

    def test_fixtures_are_in_package_data(self):
        import importlib.resources

        root = importlib.resources.files("foveate.bench")
        for name in ("finer", "formula", "ddxplus"):
            assert root.joinpath("fixtures").joinpath(f"{name}.jsonl").is_file()
            assert root.joinpath("seeds").joinpath(f"{name}.md").is_file()


class TestFiner:
    def test_row_validation(self):
        with pytest.raises(errors.ValidationError):
            Finer().row_to_sample({"tokens": ["a"]})
        with pytest.raises(errors.ValidationError):
            Finer().row_to_sample({"tokens": ["a"], "labels": ["O", "O"]})

    def test_grading_tolerates_token_prefix_and_blank_lines(self):
        f = Finer()
        assert f.is_correct("Apple\tB-ORG\n\nInc.\tI-ORG", "B-ORG\nI-ORG")
        assert not f.is_correct("B-ORG\nO", "B-ORG\nI-ORG")
        assert f.extract_answer("  B-ORG\nO ") == "B-ORG\nO"


class TestFormula:
    @pytest.mark.parametrize(
        "pred,target,ok",
        [
            ("400000", "400000", True),
            ("400000.0", "400000", True),
            ("$400,000", "400000", True),
            ("1,234,567", "1234567", True),
            ("15%", "0.15", True),
            ("15%", "15", True),
            ("0.15", "15%", True),
            ("0", "0", True),
            ("0.001", "0", True),
            ("500", "400", False),
            ("abc", "400", False),
            ("400", "abc", False),
        ],
    )
    def test_numeric_matching(self, pred, target, ok):
        assert Formula().is_correct(pred, target) is ok

    def test_zero_answer_kept_and_validation(self):
        sample = Formula().row_to_sample({"question": "q", "answer": 0})
        assert sample.target == "0"
        with pytest.raises(errors.ValidationError):
            Formula().row_to_sample({"question": "q"})
        with pytest.raises(errors.ConfigError):
            Formula(tolerance=-1)


class TestDDXPlus:
    def test_grading(self):
        d = DDXPlus()
        assert d.is_correct("2.", "2") and d.is_correct(" 2) Migraine", "2")
        assert d.is_correct("0", "0")
        assert not d.is_correct("none", "0") and not d.is_correct("1", "2")

    def test_answer_prefix_stripping(self):
        assert DDXPlus().extract_answer("\nAnswer: 3\nbecause") == "3"
        assert DDXPlus().extract_answer("") == ""


class TestRunner:
    def test_prompt_contains_playbook_only_when_given(self):
        benchmark = DDXPlus()
        sample = benchmark.load_samples(1)[0]
        plain = benchmark.build_messages(sample, None)
        armed = benchmark.build_messages(sample, benchmark.seed_playbook())
        assert len(plain) == 1 and len(armed) == 2
        assert "Playbook of strategies" in armed[0].content

    def test_backend_errors_are_not_counted_as_wrong(self):
        benchmark = Formula()
        samples = benchmark.load_samples(2)
        rt, _ = make_runtime(
            lambda r: samples[1].target,
            steps=[errors.PermanentBackendError("down")],
        )
        result = asyncio.run(Runner(rt).arun(benchmark, samples, [Arm("only")]))
        arm = result.arms[0]
        assert arm.backend_errors == 1 and arm.correct[0] is None
        assert arm.accuracy == 1.0  # scored over the 1 healthy sample

    def test_all_errors_gives_zero(self):
        benchmark = Formula()
        samples = benchmark.load_samples(2)
        rt, _ = make_runtime(steps=[errors.PermanentBackendError("down")] * 2)
        result = asyncio.run(Runner(rt).arun(benchmark, samples, [Arm("only")]))
        assert (
            result.arms[0].accuracy == 0.0
            and result.arms[0].backend_errors == 2
        )

    def test_validation(self):
        rt, _ = make_runtime()
        with pytest.raises(errors.ConfigError):
            asyncio.run(Runner(rt).arun(Formula(), [], [Arm("a")]))
