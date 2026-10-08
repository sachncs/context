import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveate import Context, Message, Role, compression, errors
from foveate import runtime as runtime_lib
from foveate.cache import base as cache_base
from foveate.compression import Budget, Overflow
from foveate.tokenizers import HeuristicTokenizer
from tests import faults as scripted


def make_runtime(responder=None, steps=(), **kw):
    backend = scripted.ScriptedBackend(
        steps, default=responder or (lambda r: "s" * 40)
    )
    rt = runtime_lib.Runtime(
        backend=backend,
        cache=cache_base.MemoryCache(),
        tokenizer=HeuristicTokenizer(),
        **kw,
    )
    return rt, backend


def doc(sentences=200):
    return " ".join(
        f"Sentence number {i} about topic {i % 7}." for i in range(sentences)
    )


def ctx_of(rt, text=None, extra=()):
    msgs = [Message(Role.SYSTEM, "be helpful"), *extra]
    msgs.append(Message(Role.USER, text if text is not None else doc()))
    return Context(tuple(msgs), rt)


class TestContext:
    def test_construction_and_roundtrip(self):
        rt, _ = make_runtime()
        c = Context.from_dicts([{"role": "user", "content": "hi there"}], rt)
        assert c.to_dicts() == [{"role": "user", "content": "hi there"}]
        assert c.token_count == 2
        assert isinstance(c.messages, tuple)

    def test_list_input_coerced_and_validated(self):
        rt, _ = make_runtime()
        c = Context([Message(Role.USER, "x")], rt)  # type: ignore[arg-type]
        assert isinstance(c.messages, tuple)
        with pytest.raises(errors.ValidationError):
            Context(({"role": "user"},), rt)  # type: ignore[arg-type]

    def test_metadata_is_read_only(self):
        rt, _ = make_runtime()
        c = Context((), rt, metadata={"a": "b"})
        with pytest.raises(TypeError):
            c.metadata["x"] = "y"  # type: ignore[index]

    def test_frozen(self):
        rt, _ = make_runtime()
        c = Context((), rt)
        with pytest.raises(AttributeError):
            c.messages = ()  # type: ignore[misc]


class TestBudgetAndReport:
    def test_within_budget_is_noop_without_llm_calls(self):
        rt, backend = make_runtime()
        c = ctx_of(rt, "short text")
        out = c.compress("ppa", budget=1000)
        assert backend.requests == []
        assert out.messages == c.messages
        assert out.report.steps == () and out.report.ratio == 1.0

    def test_invalid_budget(self):
        with pytest.raises(errors.ConfigError):
            Budget(0)

    def test_budget_enforced_by_ppa(self):
        rt, _ = make_runtime()
        out = ctx_of(rt).compress("ppa", budget=300, leaf_tokens=128)
        assert out.token_count <= 300
        assert out.report.final_tokens == out.token_count
        assert out.report.original_tokens > 300
        assert out.report.method == "ppa" and not out.report.truncated

    def test_overflow_raise_vs_truncate(self):
        rt, _ = make_runtime(lambda r: "word " * 400)
        c = ctx_of(rt)
        with pytest.raises(errors.BudgetExceededError) as info:
            c.compress("ppa", budget=100)
        assert info.value.budget == 100 and info.value.actual > 100
        out = c.compress("ppa", budget=Budget(100, Overflow.TRUNCATE))
        assert out.token_count <= 100 and out.report.truncated

    def test_unreachable_budget_raises_even_with_truncate(self):
        rt, _ = make_runtime()
        c = Context((Message(Role.SYSTEM, "x" * 400),), rt)
        with pytest.raises(errors.BudgetExceededError):
            c.compress("truncate", budget=Budget(1, Overflow.RAISE))

    def test_report_usage_and_cost(self):
        from foveate import usage

        rt, _ = make_runtime(
            prices=usage.PriceTable({"gpt-4o-mini": usage.Price(1e6, 1e6)})
        )
        out = ctx_of(rt).compress("ppa", budget=300, leaf_tokens=128)
        assert out.report.usage.total_tokens > 0
        assert out.report.cost_usd == pytest.approx(
            out.report.usage.total_tokens
        )
        assert out.report.seconds >= 0


class TestPpa:
    def test_single_leaf(self):
        rt, backend = make_runtime()
        c = ctx_of(rt, "word " * 400)
        out = c.compress("ppa", budget=100, leaf_tokens=1000)
        assert len(backend.requests) == 1
        assert out.messages[0].content == "be helpful"
        assert out.messages[-1].content == "s" * 40

    def test_multi_leaf_order_and_combine_if_needed(self):
        seen = []

        def responder(request):
            seen.append(request.messages[-1].content)
            return f"S{len(seen)}"

        rt, backend = make_runtime(responder)
        out = ctx_of(rt).compress("ppa", budget=400, leaf_tokens=128)
        names = [s.name for s in out.report.steps]
        assert "combine" not in names  # joined summaries already fit
        assert sum(n.startswith("leaf") for n in names) > 3

    def test_combine_always_and_never(self):
        rt, _ = make_runtime()
        always = ctx_of(rt).compress(
            "ppa", budget=400, leaf_tokens=128, combine="always"
        )
        assert any(s.name == "combine" for s in always.report.steps)
        never = ctx_of(rt).compress(
            "ppa", budget=400, leaf_tokens=128, combine="never"
        )
        assert not any(s.name == "combine" for s in never.report.steps)

    def test_combine_runs_when_summaries_too_big(self):
        calls = {"combine": 0}

        def responder(request):
            if "section summaries" in request.messages[-1].content:
                calls["combine"] += 1
                return "tiny"
            return "x" * 200

        rt, _ = make_runtime(responder)
        out = ctx_of(rt).compress("ppa", budget=200, leaf_tokens=128)
        assert calls["combine"] == 1 and out.token_count <= 200

    def test_leaf_failure_wrapped_and_progress_cached(self):
        calls = {"n": 0}

        def responder(request):
            calls["n"] += 1
            return "s" * 40

        rt, backend = make_runtime(responder)
        text = doc()
        # the 3rd backend call fails permanently
        backend.pending.extend(
            ["s" * 40, "s" * 40, errors.PermanentBackendError("boom")]
        )
        c = ctx_of(rt, text)
        with pytest.raises(errors.CompressionError) as info:
            c.compress("ppa", budget=300, leaf_tokens=128)
        assert info.value.step.startswith("leaf")
        assert isinstance(info.value.cause, errors.PermanentBackendError)
        before = len(backend.requests)
        out = c.compress("ppa", budget=300, leaf_tokens=128)
        assert out.token_count <= 300
        # the retry reused whatever finished before the failure
        assert out.report.cache_hits >= 1
        assert len(backend.requests) > before

    def test_max_leaves(self):
        rt, _ = make_runtime()
        with pytest.raises(errors.CompressionError, match="max_leaves"):
            ctx_of(rt).compress("ppa", budget=100, leaf_tokens=16, max_leaves=2)

    def test_prompt_change_invalidates_cache(self):
        from foveate.compression import ppa

        rt, backend = make_runtime()
        c = ctx_of(rt, "word " * 400)
        c.compress("ppa", budget=100, leaf_tokens=1000)
        c.compress("ppa", budget=100, leaf_tokens=1000)
        assert len(backend.requests) == 1
        original = ppa.SUMMARIZE
        try:
            ppa.SUMMARIZE = original.__class__(
                original.name, "3", original.system, original.user
            )
            c.compress("ppa", budget=100, leaf_tokens=1000)
        finally:
            ppa.SUMMARIZE = original
        assert len(backend.requests) == 2

    def test_other_messages_exceed_budget_raises(self):
        rt, _ = make_runtime()
        c = Context(
            (
                Message(Role.SYSTEM, "x" * 4000),
                Message(Role.USER, "y" * 200),
            ),
            rt,
        )
        with pytest.raises(errors.BudgetExceededError):
            c.compress("ppa", budget=500)

    def test_option_validation(self):
        for bad in (
            {"leaf_tokens": 1},
            {"min_summary_tokens": 0},
            {"max_leaves": 0},
        ):
            with pytest.raises(errors.ConfigError):
                compression.resolve("ppa", **bad)


class TestHierarchical:
    def test_multiple_rounds_reach_budget(self):
        sizes = iter([2000, 400, 60])

        def responder(request):
            return "w" * next(sizes, 40)

        rt, _ = make_runtime(responder)
        out = ctx_of(rt, "word " * 4000).compress(
            "hierarchical", budget=100, leaf_tokens=4000
        )
        assert out.token_count <= 100
        assert len(out.report.steps) >= 3

    def test_stops_without_progress(self):
        rt, _ = make_runtime(lambda r: "w" * 4000)
        with pytest.raises(errors.BudgetExceededError):
            ctx_of(rt, "word " * 4000).compress(
                "hierarchical", budget=100, max_rounds=5
            )

    def test_validation(self):
        with pytest.raises(errors.ConfigError):
            compression.resolve("hierarchical", max_rounds=0)


class TestUShape:
    def convo(self, rt, n=12):
        return Context(
            tuple(
                Message(
                    Role.USER if i % 2 == 0 else Role.ASSISTANT,
                    f"m{i} " + "w " * 80,
                )
                for i in range(n)
            ),
            rt,
        )

    def test_summarize_middle(self):
        rt, _ = make_runtime()
        c = self.convo(rt)
        out = c.compress("ushape", budget=400, head=2, tail=3)
        assert out.messages[:2] == c.messages[:2]
        assert out.messages[-3:] == c.messages[-3:]
        assert len(out.messages) == 6
        assert out.messages[2].content.startswith("[Summary of earlier")
        assert out.report.final_tokens > 0 and out.report.original_tokens > 0
        assert len(out.report.steps) >= 1

    def test_drop_middle(self):
        rt, backend = make_runtime()
        out = self.convo(rt).compress(
            "ushape", budget=300, head=1, tail=1, middle="drop"
        )
        assert backend.requests == []
        assert out.messages[1].content == "[10 earlier messages omitted]"

    def test_nothing_to_compress(self):
        rt, _ = make_runtime()
        c = self.convo(rt, 4)
        with pytest.raises(errors.BudgetExceededError):
            c.compress("ushape", budget=10, head=2, tail=2)

    def test_validation(self):
        with pytest.raises(errors.ConfigError):
            compression.resolve("ushape", head=-1)

    def test_tail_zero(self):
        rt, _ = make_runtime()
        out = self.convo(rt).compress("ushape", budget=300, head=1, tail=0)
        assert len(out.messages) == 2


class TestLlmFreeStrategies:
    def test_truncate_sides(self):
        rt, backend = make_runtime()
        text = "".join(f"line{i:03d}\n" for i in range(300))
        for side in ("head", "tail", "middle"):
            out = ctx_of(rt, text).compress("truncate", budget=100, keep=side)
            assert out.token_count <= 100
            body = out.messages[-1].content
            assert "[... truncated ...]" in body
            if side == "head":
                assert body.startswith("line000")
            if side == "tail":
                assert body.rstrip().endswith("line299")
            if side == "middle":
                assert body.startswith("line000") and body.rstrip().endswith(
                    "line299"
                )
        assert backend.requests == []

    def test_window_drops_oldest_keeps_system(self):
        rt, _ = make_runtime()
        msgs = [Message(Role.SYSTEM, "sys")] + [
            Message(Role.USER, f"{i} " + "w " * 40) for i in range(10)
        ]
        out = Context(tuple(msgs), rt).compress("window", budget=120)
        assert out.messages[0].content == "sys"
        assert out.messages[-1] == msgs[-1]
        assert len(out.messages) < len(msgs)
        assert out.token_count <= 120

    def test_window_min_messages_protected(self):
        rt, _ = make_runtime()
        msgs = [Message(Role.USER, "w " * 100) for _ in range(5)]
        with pytest.raises(errors.BudgetExceededError):
            Context(tuple(msgs), rt).compress(
                "window", budget=10, min_messages=5
            )
        with pytest.raises(errors.ConfigError):
            compression.resolve("window", min_messages=-1)

    def test_extractive_keeps_order_and_budget(self):
        rt, backend = make_runtime()
        out = ctx_of(rt).compress("extractive", budget=150)
        assert out.token_count <= 150 and backend.requests == []
        body = out.messages[-1].content
        numbers = [
            int(s.split()[2])
            for s in body.split(". ")
            if s.startswith("Sentence")
        ]
        assert numbers == sorted(numbers)

    def test_extractive_single_sentence_unchanged_then_enforced(self):
        rt, _ = make_runtime()
        c = ctx_of(rt, "x" * 2000)
        out = c.compress("extractive", budget=Budget(100, Overflow.TRUNCATE))
        assert out.token_count <= 100
        with pytest.raises(errors.ConfigError):
            compression.resolve("extractive", edge_bonus=-1)
        with pytest.raises(errors.ConfigError):
            compression.resolve("extractive", unit_tokens=0)

    def test_extractive_reduces_text_without_sentence_punctuation(self):
        rt, _ = make_runtime()
        c = Context(
            (Message(Role.USER, " ".join(f"tok{i}" for i in range(900))),), rt
        )
        out = c.compress("extractive", budget=200)
        assert 0 < out.token_count <= 200


class TestComposition:
    def test_pipeline_stops_when_fits_and_labels(self):
        rt, backend = make_runtime()
        out = ctx_of(rt).compress("truncate+ppa", budget=500)
        assert out.report.method == "truncate+ppa"
        assert backend.requests == []  # truncate alone fit the budget
        assert out.token_count <= 500

    def test_pipeline_options_per_stage(self):
        rt, _ = make_runtime()
        out = ctx_of(rt).compress(
            "window+ppa", budget=300, ppa={"leaf_tokens": 128}
        )
        assert out.token_count <= 300

    def test_fallback_on_backend_failure(self):
        rt, backend = make_runtime(steps=[errors.PermanentBackendError("down")])
        out = ctx_of(rt).compress("ppa|extractive", budget=200)
        assert out.token_count <= 200
        assert out.report.method == "ppa|extractive"
        assert any(
            s.name.startswith("fallback after") for s in out.report.steps
        )

    def test_fallback_not_used_on_success(self):
        rt, _ = make_runtime()
        out = ctx_of(rt, "word " * 400).compress("ppa|extractive", budget=100)
        assert not any(s.name.startswith("fallback") for s in out.report.steps)
        assert out.messages[-1].content == "s" * 40

    def test_instance_method_and_errors(self):
        rt, _ = make_runtime()
        strategy = compression.Truncate(keep=compression.Side.HEAD)
        out = ctx_of(rt).compress(strategy, budget=100)
        assert out.token_count <= 100
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).compress(strategy, budget=100, keep="tail")
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).compress("nope", budget=100)
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).compress("ppa", budget=100, bogus=1)
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).compress("window+ppa", budget=100, truncate={})
        with pytest.raises(errors.ConfigError):
            ctx_of(rt).compress("window+ppa", budget=100, ppa=5)
        with pytest.raises(errors.ConfigError):
            compression.Pipeline(())

    def test_acompress_and_sync_inside_loop(self):
        rt, _ = make_runtime()
        c = ctx_of(rt)

        async def go():
            a = await c.acompress("truncate", budget=100)
            b = c.compress("truncate", budget=100)  # sync call in a loop
            return a, b

        a, b = asyncio.run(go())
        assert a.messages == b.messages


@settings(max_examples=40, deadline=None)
@given(
    text=st.text(alphabet="abc .\n", min_size=50, max_size=1500),
    budget=st.integers(min_value=20, max_value=200),
)
def test_pipeline_output_within_budget(text, budget):
    rt, _ = make_runtime()
    c = Context((Message(Role.USER, text),), rt)
    out = c.compress(
        "extractive+truncate", budget=Budget(budget, Overflow.TRUNCATE)
    )
    assert out.token_count <= budget


class TestFairAllocation:
    def conversation(self, rt, count=20, words=120):
        return Context(
            tuple(
                Message(Role.USER, f"turn {i}: " + "word " * words)
                for i in range(count)
            ),
            rt,
        )

    def test_many_midsize_messages_reach_budget_with_ppa(self):
        rt, backend = make_runtime(lambda r: "short summary")
        out = self.conversation(rt).compress("ppa", budget=400)
        assert out.token_count <= 400
        assert len(out.messages) == 20  # every turn kept, each compressed
        assert any(s.name.startswith("m0 ") for s in out.report.steps)

    def test_small_messages_are_left_alone(self):
        rt, _ = make_runtime(lambda r: "s" * 40)
        small = Message(Role.USER, "tiny note")
        big = Message(Role.USER, "word " * 2000)
        out = Context((small, big), rt).compress("ppa", budget=300)
        assert out.messages[0] == small and out.token_count <= 300

    def test_extractive_handles_many_messages(self):
        rt, backend = make_runtime()
        convo = Context(
            tuple(
                Message(
                    Role.USER,
                    " ".join(
                        f"Sentence {j} of turn {i} about topic {j % 3}."
                        for j in range(30)
                    ),
                )
                for i in range(10)
            ),
            rt,
        )
        out = convo.compress("extractive", budget=300)
        assert out.token_count <= 300 and backend.requests == []

    def test_system_messages_are_protected(self):
        rt, _ = make_runtime(lambda r: "s" * 20)
        system = Message(Role.SYSTEM, "keep me " * 20)
        out = Context(
            (system, Message(Role.USER, "word " * 1000)), rt
        ).compress("ppa", budget=300)
        assert out.messages[0] == system

    def test_fallback_conversation_example(self):
        rt, _ = make_runtime(steps=[errors.PermanentBackendError("outage")])
        out = self.conversation(rt).compress("ppa|extractive", budget=900)
        assert out.token_count <= 900


@settings(max_examples=100, deadline=None)
@given(
    sizes=st.lists(
        st.integers(min_value=1, max_value=500), min_size=1, max_size=15
    ),
    budget=st.integers(min_value=1, max_value=3000),
)
def test_fair_targets_properties(sizes, budget):
    from foveate.compression import fit

    tok = HeuristicTokenizer(chars_per_token=1.0)
    messages = tuple(Message(Role.USER, "x" * n) for n in sizes)
    targets = fit.fair_targets(messages, budget, tok)
    after = [targets.get(i, n) for i, n in enumerate(sizes)]
    assert all(t < sizes[i] for i, t in targets.items())  # only real shrinks
    if sum(sizes) <= budget:
        assert targets == {}
    elif budget >= len(sizes):
        assert sum(after) <= budget  # fits whenever the floor allows it
        cap = max(targets.values())
        assert all(n <= cap for i, n in enumerate(sizes) if i not in targets)


class TestShortenPass:
    def test_overlong_summary_is_rewritten_shorter(self):
        calls = []

        def responder(request):
            text = request.messages[-1].content
            calls.append(
                "shorten" if "Rewrite the text below" in text else "summarise"
            )
            return "w " * 400 if calls[-1] == "summarise" else "short answer"

        rt, backend = make_runtime(responder)
        out = ctx_of(rt, "word " * 2000).compress(
            "ppa", budget=120, leaf_tokens=4000
        )
        assert calls == ["summarise", "shorten"]
        assert (
            out.messages[-1].content == "short answer"
            and out.token_count <= 120
        )
        assert [s.name for s in out.report.steps] == [
            "leaf 0",
            "leaf 0 shorten 1",
        ]

    def test_shorten_is_bounded(self):
        counter = iter(range(1000))
        rt, backend = make_runtime(lambda r: "w " * (400 + next(counter)))
        with pytest.raises(errors.BudgetExceededError):
            ctx_of(rt, "word " * 2000).compress(
                "ppa", budget=120, leaf_tokens=4000
            )
        assert len(backend.requests) == 1 + 2  # one summary + SHORTEN_ROUNDS

    def test_within_limit_summary_is_left_alone(self):
        rt, backend = make_runtime(lambda r: "w " * 60)
        ctx_of(rt, "word " * 2000).compress("ppa", budget=120, leaf_tokens=4000)
        assert len(backend.requests) == 1
