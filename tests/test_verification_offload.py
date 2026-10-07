import pytest

from ceng import Context, Message, Role, errors, verification
from ceng.compression import Offload
from ceng.stores import MemoryNotesStore
from ceng.verification import macro
from tests.test_compression import ctx_of, make_runtime

TREE = [
    {
        "description": "young",
        "prior": 0.25,
    },
    {
        "description": "old",
        "prior": 0.75,
        "children": [
            {"description": "old-a", "prior": 0.5},
            {"description": "old-b", "prior": 0.5},
        ],
    },
]


def answers(direct, leaves):
    table = {"young": leaves[0], "old-a": leaves[1], "old-b": leaves[2]}

    def respond(request):
        text = request.messages[-1].content
        for name, value in table.items():
            if f"subpopulation: {name}\n" in text:
                return str(value)
        return str(direct)

    return respond


class TestParseProbability:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0.5", 0.5),
            ("50%", 0.5),
            ("About 0.25 maybe", 0.25),
            (".3", 0.3),
            ("5e-1", 0.5),
            ("1", 1.0),
            ("0", 0.0),
            ("100 %", 1.0),
        ],
    )
    def test_ok(self, raw, expected):
        assert macro.parse_probability(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["none", "", "75", "-0.1", "1.5"])
    def test_rejected(self, raw):
        with pytest.raises(errors.ValidationError):
            macro.parse_probability(raw)


class TestTree:
    def test_build_and_leaves(self):
        roots = macro.TreeNode.build(TREE)
        leaves = [(d, round(p, 4)) for r in roots for d, p in r.leaves()]
        assert leaves == [("young", 0.25), ("old-a", 0.375), ("old-b", 0.375)]

    @pytest.mark.parametrize(
        "raw",
        [
            [{"description": "a", "prior": 0.5}],  # roots sum != 1
            [
                {
                    "description": "a",
                    "children": [{"description": "b", "prior": 0.4}],
                }
            ],
            [{"description": ""}],
            [{"prior": 1}],
            [{"description": "a", "prior": 0}],
            [{"description": "a", "prior": 2}],
            [{"description": "a", "prior": "x"}],
            [{"description": "a", "prior": True}],
            ["not a mapping"],
            [{"description": "a", "children": "oops"}],
        ],
    )
    def test_invalid(self, raw):
        with pytest.raises(errors.ValidationError):
            macro.TreeNode.build(raw)

    def test_very_deep_tree_does_not_overflow_stack(self):
        node = {"description": "leaf"}
        for _ in range(3000):
            node = {"description": "n", "children": [node]}
        roots = macro.TreeNode.build([node])
        assert roots[0].leaves()[0][0] == "leaf"


class TestMacroFallacy:
    def run(self, direct, leaves, **kw):
        rt, backend = make_runtime(answers(direct, leaves))
        ctx = Context((Message(Role.USER, "x"),), rt)
        verdict = ctx.verify(
            "macro_fallacy",
            question="Do they like it?",
            population="users",
            tree=TREE,
            **kw,
        )
        return verdict, backend

    def test_consistent(self):
        verdict, backend = self.run(0.5, [0.4, 0.5, 0.5])
        assert verdict.passed and verdict.verifier == "macro_fallacy"
        assert verdict.aggregated_estimate == pytest.approx(0.1 + 0.75 * 0.5)
        assert len(verdict.leaves) == 3 and len(backend.requests) == 4

    def test_fallacy_detected(self):
        verdict, _ = self.run(0.9, [0.1, 0.1, 0.1])
        assert not verdict.passed and verdict.delta > verdict.tolerance

    def test_tolerance_and_cache(self):
        verdict, backend = self.run(0.9, [0.1, 0.1, 0.1], tolerance=1.0)
        assert verdict.passed

    def test_second_run_is_cached(self):
        rt, backend = make_runtime(answers(0.5, [0.5, 0.5, 0.5]))
        ctx = Context((Message(Role.USER, "x"),), rt)
        options = {"question": "q", "population": "p", "tree": TREE}
        ctx.verify("macro_fallacy", **options)
        second = ctx.verify("macro_fallacy", **options)
        assert len(backend.requests) == 4
        assert all(leaf.cached for leaf in second.leaves)
        assert second.population_cached

    def test_garbage_answer_wrapped(self):
        rt, _ = make_runtime(lambda r: "no idea")
        ctx = Context((Message(Role.USER, "x"),), rt)
        with pytest.raises(errors.CompressionError):
            ctx.verify("macro_fallacy", question="q", population="p", tree=TREE)

    def test_backend_failure_wrapped(self):
        rt, _ = make_runtime(steps=[errors.PermanentBackendError("down")])
        ctx = Context((Message(Role.USER, "x"),), rt)
        with pytest.raises(errors.CompressionError):
            ctx.verify("macro_fallacy", question="q", population="p", tree=TREE)

    def test_config_validation(self):
        roots = macro.TreeNode.build(TREE)
        with pytest.raises(errors.ConfigError):
            macro.MacroFallacy("q", "p", roots, tolerance=-1)
        with pytest.raises(errors.ConfigError):
            macro.MacroFallacy("q", "p", ())
        with pytest.raises(errors.ConfigError):
            macro.MacroFallacy("q", "p", [roots[0]])  # type: ignore[arg-type]
        with pytest.raises(errors.ConfigError):
            verification.resolve("macro_fallacy", bogus=1)
        with pytest.raises(errors.ConfigError):
            verification.resolve("nope")


class TestFitsAndResolve:
    def test_fits(self):
        rt, _ = make_runtime()
        ctx = ctx_of(rt)
        assert not ctx.verify("fits", tokens=10).passed
        assert ctx.verify("fits", tokens=100000).passed
        with pytest.raises(errors.ConfigError):
            ctx.verify("fits", tokens=0)

    def test_instance_resolution(self):
        rt, _ = make_runtime()
        instance = verification.FitsBudget(10_000)
        assert ctx_of(rt).verify(instance).passed
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).verify(instance, tokens=1)


class TestOffload:
    def convo(self, rt, n=10):
        return Context(
            tuple(Message(Role.USER, f"m{i} " + "w " * 60) for i in range(n)),
            rt,
        )

    def test_offloads_and_is_lossless(self):
        rt, backend = make_runtime()
        store = MemoryNotesStore()
        out = self.convo(rt).compress(
            "offload", budget=250, store=store, tail=2
        )
        assert backend.requests == []
        assert len(out.messages) == 1 + 1 + 2
        pointer = out.messages[1].content
        path = pointer.split("note ")[1].rstrip("]")
        saved = store.read(path)
        assert "m1 " in saved.content and "m7 " in saved.content
        assert saved.tags == ("offload",)

    def test_idempotent(self):
        rt, _ = make_runtime()
        store = MemoryNotesStore()
        c = self.convo(rt)
        c.compress("offload", budget=250, store=store, tail=2)
        c.compress("offload", budget=250, store=store, tail=2)
        assert len(store.entries()) == 1

    def test_nothing_to_offload_and_validation(self):
        rt, _ = make_runtime()
        store = MemoryNotesStore()
        small = self.convo(rt, 3)
        with pytest.raises(errors.BudgetExceededError):
            small.compress("offload", budget=5, store=store, head=1, tail=2)
        with pytest.raises(errors.ConfigError):
            Offload(store, head=-1)
        out = self.convo(rt).compress(
            "offload", budget=250, store=store, head=1, tail=0
        )
        assert len(out.messages) == 2
