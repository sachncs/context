"""Position-aware order, query-aware condensing, clearing, support, caching."""

import asyncio
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from foveate import Budget, Context, Foveator, Message, Overflow, Role, errors
from foveate.backends import providers
from foveate.compression import ClearToolResults, QueryExtractive
from foveate.foveation import (
    FoveationConfig,
    PagePlan,
    Tier,
    allocate,
    arrange,
)
from foveate.tokenizers import HeuristicTokenizer
from foveate.usage import Usage
from tests.test_compression import make_runtime
from tests.test_longdoc import make_doc, rt

TOK = HeuristicTokenizer()


def plan(page, score, tier=Tier.FULL):
    return PagePlan("d", page, tier, score, 10, 10, "t")


class TestArrange:
    def test_reading_order_is_unchanged(self):
        pages = [plan(1, 0.1), plan(2, 0.9), plan(3, 0.5)]
        assert arrange(pages, "reading") == pages

    def test_edges_put_the_best_pages_at_both_ends(self):
        pages = [
            plan(n, s)
            for n, s in [(1, 0.1), (2, 0.9), (3, 0.5), (4, 0.7), (5, 0.3)]
        ]
        order = [p.page for p in arrange(pages, "edges")]
        assert order[0] == 2 and order[-1] == 4  # best first, second best last
        assert order[len(order) // 2] == 1  # the weakest sits in the middle
        assert sorted(order) == [1, 2, 3, 4, 5]

    def test_condensed_pages_stay_in_the_middle(self):
        pages = [plan(1, 0.9), plan(2, 0.0, Tier.CONDENSED), plan(3, 0.8)]
        assert [p.page for p in arrange(pages, "edges")] == [1, 2, 3]

    @given(st.lists(st.floats(0, 1), min_size=1, max_size=20, unique=True))
    def test_edges_is_a_permutation_with_the_top_page_first(self, scores):
        pages = [plan(i + 1, s) for i, s in enumerate(scores)]
        order = arrange(pages, "edges")
        assert sorted(p.page for p in order) == sorted(p.page for p in pages)
        assert order[0].score == max(scores)

    def test_unknown_order_is_rejected(self):
        with pytest.raises(errors.ConfigError):
            FoveationConfig(order="sideways")

    def test_foveator_renders_in_the_configured_order(self):
        doc = make_doc(pages=30)
        foveator = Foveator(
            rt(), budget=4000, foveation=FoveationConfig(order="edges")
        )
        _, pages = foveator.render(
            foveator.plan("capital expenditures fiscal 2018", [doc]).foveation
        )
        assert pages.index("p.7]") < pages.index("p.6 |")  # best page first


class TestQueryAware:
    TEXT = (
        "The committee met on Tuesday to review parking. "
        "Catering was discussed at length. "
        "Capital expenditures were 1,577 million dollars in fiscal 2018. "
        "The meeting ended at noon."
    )

    def test_keeps_the_sentence_that_matches_the_query(self):
        q = QueryExtractive(query="capital expenditures fiscal 2018")
        kept = q.extract(self.TEXT, 22, TOK)
        assert "1,577" in kept and "parking" not in kept

    def test_plain_extractive_may_drop_it(self):
        from foveate.compression import Extractive

        # frequency scoring has no idea what the reader wants
        kept = Extractive().extract(self.TEXT, 22, TOK)
        assert "1,577" not in kept or "parking" in kept

    def test_without_a_query_it_falls_back_to_frequency(self):
        assert QueryExtractive().extract(self.TEXT, 22, TOK)

    def test_compress_uses_the_last_user_message_as_the_query(self):
        runtime, backend = make_runtime()
        ctx = Context(
            [
                Message(Role.USER, self.TEXT * 6),
                Message(Role.USER, "capital expenditures fiscal 2018?"),
            ],
            runtime=runtime,
        )
        out = ctx.compress("query", budget=60)
        assert "1,577" in out.messages[0].content and backend.requests == []

    def test_needs_a_query_somewhere(self):
        runtime, _ = make_runtime()
        ctx = Context([Message(Role.ASSISTANT, "word " * 400)], runtime=runtime)
        with pytest.raises(errors.ConfigError):
            ctx.compress("query", budget=50)

    def test_allocation_condenses_neighbours_around_the_question(self):
        doc = make_doc(
            pages=5,
            facts={
                3: "Capital expenditures were 1,577 million dollars in fiscal 2018."
            },
        )
        ranked = [(("acme_2018", 2), 1.0)]
        cfg = FoveationConfig(query_aware=True, condensed_tokens=40)
        fov = allocate(
            ranked, [doc], 1500, TOK, cfg, query="capital expenditures 1,577"
        )
        neighbour = next(p for p in fov.pages if p.page == 3)
        assert neighbour.tier is Tier.CONDENSED and "1,577" in neighbour.text


class TestClearToolResults:
    def conversation(self):
        msgs = [Message(Role.USER, "go")]
        for i in range(5):
            msgs.append(Message(Role.ASSISTANT, f"calling tool {i}"))
            msgs.append(
                Message(
                    Role.TOOL, f"result {i} " + "data " * 120, name=f"lookup{i}"
                )
            )
        msgs.append(
            Message(Role.ASSISTANT, "[tool result flat: " + "x " * 200 + "]")
        )
        return msgs

    def test_clears_oldest_results_first_until_it_fits(self):
        runtime, backend = make_runtime()
        ctx = Context(self.conversation(), runtime=runtime)
        out = ctx.compress("clear_tool_results", budget=ctx.token_count - 200)
        cleared = [m for m in out.messages if "cleared" in m.content]
        assert cleared and "lookup0" in cleared[0].content
        assert out.messages[-1].content.startswith("[tool result flat")
        assert backend.requests == []
        assert out.report.final_tokens <= ctx.token_count - 200

    def test_newest_results_are_kept(self):
        runtime, _ = make_runtime()
        ctx = Context(self.conversation(), runtime=runtime)
        result = asyncio.run(
            ClearToolResults(keep=3).compress(
                ctx, Budget(300, Overflow.TRUNCATE)
            )
        )
        tools = [m.content for m in result.messages if m.role is Role.TOOL]
        assert sum("cleared" in c for c in tools) == 3  # 6 results, keep 3

    def test_negative_keep_is_rejected(self):
        with pytest.raises(errors.ConfigError):
            ClearToolResults(keep=-1)


class TestSupportAndCaching:
    def test_support_is_the_share_of_verified_quotes(self):
        doc = make_doc(pages=10)

        def reply(request):
            return json.dumps(
                {
                    "found": True,
                    "answer": "1,577 million",
                    "citations": [
                        {
                            "doc": "acme_2018",
                            "page": 7,
                            "quote": "Capital expenditures were 1,577 million dollars",
                        },
                        {
                            "doc": "acme_2018",
                            "page": 3,
                            "quote": "a quote that is not on page three",
                        },
                    ],
                }
            )

        from foveate import runtime as runtime_lib
        from foveate.cache import base as cache_base
        from tests import faults

        runtime = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(default=reply),
            cache=cache_base.MemoryCache(),
            tokenizer=TOK,
        )
        answer = Foveator(runtime, budget=4000, max_rounds=1).ask("q?", [doc])
        assert answer.support == 0.5 and not answer.grounded

    def test_support_is_zero_without_citations(self):
        answer = Foveator(rt(), budget=4000).ask("zzz unrelated?", [make_doc()])
        assert answer.support == 0.0

    def test_question_comes_last_in_the_prompt_so_the_prefix_is_stable(self):
        from foveate import runtime as runtime_lib
        from tests import faults

        seen = []

        def reply(request):
            seen.append(request.messages[-1].content)
            return json.dumps({"found": False, "answer": "", "citations": []})

        runtime = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(default=reply), tokenizer=TOK
        )
        foveator = Foveator(runtime, budget=4000, max_rounds=1)
        foveator.ask("first question about capex?", [make_doc()])
        assert (
            seen[0].rstrip().endswith("Question: first question about capex?")
        )

    def test_usage_adds_cached_tokens(self):
        assert (Usage(10, 2, 4) + Usage(5, 1, 1)).cached_tokens == 5

    def test_cached_tokens_are_read_from_openai_shaped_responses(self):
        response = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 128},
            }
        }
        assert providers.usage_from_response(response).cached_tokens == 128
        bare = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        assert providers.usage_from_response(bare).cached_tokens == 0
        assert providers.usage_from_response({}).prompt_tokens == 0


def test_inference_mode_changes_the_system_prompt_only():
    from foveate.foveator import ANSWER, ANSWER_INFERENCE

    assert "inference" in ANSWER_INFERENCE.system
    assert "inference" not in ANSWER.system
    assert "short inference" in ANSWER_INFERENCE.user
    assert ANSWER_INFERENCE.fingerprint != ANSWER.fingerprint
    seen = []

    def reply(request):
        seen.append(request.messages[0].content)
        return json.dumps({"found": False, "answer": "", "citations": []})

    from foveate import runtime as runtime_lib
    from tests import faults

    runtime = runtime_lib.Runtime(
        backend=faults.ScriptedBackend(default=reply), tokenizer=TOK
    )
    Foveator(runtime, budget=4000, max_rounds=1, inference=True).ask(
        "q?", [make_doc()]
    )
    Foveator(runtime, budget=4000, max_rounds=1).ask("q?", [make_doc()])
    assert "inference" in seen[0] and "inference" not in seen[1]


class TestDefensiveClearing:
    def convo(self):
        return [
            Message(Role.USER, "go"),
            Message(
                Role.TOOL,
                "Traceback (most recent call last): boom " + "x " * 200,
                name="run",
            ),
            Message(Role.TOOL, "result a " + "data " * 150, name="search"),
            Message(Role.TOOL, "result b " + "data " * 150, name="keepme"),
            Message(Role.TOOL, "result c " + "data " * 150, name="search"),
            Message(Role.TOOL, "result d " + "data " * 150, name="search"),
        ]

    def clear(self, **kw):
        runtime, _ = make_runtime()
        ctx = Context(self.convo(), runtime=runtime)
        out = asyncio.run(
            ClearToolResults(keep=1, **kw).compress(
                ctx, Budget(560, Overflow.TRUNCATE)
            )
        )
        return [m.content for m in out.messages]

    def test_errors_are_kept_by_default(self):
        texts = self.clear()
        assert texts[1].startswith("Traceback") and "cleared" in texts[2]

    def test_errors_can_be_cleared_when_asked(self):
        texts = self.clear(keep_errors=False)
        assert "cleared" in texts[1]

    def test_excluded_tools_are_never_cleared(self):
        texts = self.clear(exclude=("keepme",))
        assert texts[3].startswith("result b") and "cleared" in texts[2]


def test_cached_tokens_use_the_cached_price():
    from foveate.usage import Price, PriceTable

    table = PriceTable({"m": Price(3.0, 15.0, 0.3)})
    full = table.cost("m", Usage(1_000_000, 0))
    cached = table.cost("m", Usage(1_000_000, 0, 1_000_000))
    assert full == 3.0 and cached == pytest.approx(0.3)
    plain = PriceTable({"m": Price(3.0, 15.0)})
    assert plain.cost("m", Usage(1_000_000, 0, 1_000_000)) == 3.0


def test_history_trigger_and_target_leave_headroom():
    from foveate.agents import history
    from tests.test_agents_history import ToyAdapter, conversation

    runtime, _ = make_runtime()
    items = conversation(6, 80)
    eager = history.HistoryCompressor(
        ToyAdapter(),
        runtime,
        budget=1200,
        method="extractive",
        trigger=0.5,
        target=0.4,
    )
    lazy = history.HistoryCompressor(
        ToyAdapter(), runtime, budget=1200, method="extractive"
    )
    assert eager.count(items) > 1200
    out_eager = asyncio.run(eager.compress(items))
    assert (
        eager.count(out_eager) <= 1200 * 0.4 + 400
    )  # leaves headroom (tail is verbatim)
    assert (
        eager.count(out_eager)
        < lazy.count(asyncio.run(lazy.compress(items))) + 1
    )
    # under the trigger nothing happens
    small = conversation(1, 20)
    assert asyncio.run(eager.compress(small)) == small
    with pytest.raises(errors.ConfigError):
        history.HistoryCompressor(
            ToyAdapter(), runtime, budget=100, trigger=0.5, target=0.9
        )
