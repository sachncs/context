import json

import pytest

from foveate import Foveator, errors
from foveate import runtime as runtime_lib
from foveate.cache import base as cache_base
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.grounding import NOT_FOUND, normalise, supports
from foveate.tokenizers import HeuristicTokenizer
from tests import faults

TOK = HeuristicTokenizer()
FACTS = {
    7: "Capital expenditures for fiscal 2018 were 1,577 million dollars.",
    12: "The dividend declared was 1.50 per share in March.",
    19: "Headcount grew to 91,000 employees worldwide.",
}


def build_document(pages=30):
    items = []
    for n in range(1, pages + 1):
        body = (
            FACTS.get(n, f"Routine discussion of operations on page {n}. ")
            + " Filler sentence. " * 40
        )
        items.append(page_lib.Page(n, body, TOK.count(body), (f"SECTION {n}",)))
    return Document("annual", tuple(items))


def reply(**fields):
    base = {"found": True, "answer": "", "citations": [], "need_pages": []}
    return json.dumps({**base, **fields})


def foveator(replies, **kw):
    backend = faults.ScriptedBackend(replies)
    rt = runtime_lib.Runtime(
        backend=backend, cache=cache_base.MemoryCache(), tokenizer=TOK
    )
    kw.setdefault("budget", 3000)
    return Foveator(rt, **kw), backend


class TestQuotes:
    def test_normalise_numbers_and_punctuation(self):
        assert normalise("$1,577.00!") == normalise("1577.00")
        assert supports("were 1,577 million", FACTS[7])
        assert supports(
            "capital expenditures for fiscal 2018 were 1577 million", FACTS[7]
        )

    def test_rejects_invented_text(self):
        assert not supports("capital expenditures were 9,999 million", FACTS[7])
        assert not supports("", FACTS[7])
        assert not supports("42", FACTS[7])

    def test_tolerates_small_edits_in_long_quotes(self):
        quote = "Capital expenditures for fiscal 2018 were roughly 1,577 million dollars"
        assert supports(quote, FACTS[7])


class TestAsk:
    def test_grounded_answer_with_verified_citation(self):
        f, backend = foveator(
            [
                reply(
                    answer="1,577 million",
                    citations=[
                        {
                            "doc": "annual",
                            "page": 7,
                            "quote": "were 1,577 million dollars",
                        }
                    ],
                )
            ]
        )
        a = f.ask(
            "What were capital expenditures in fiscal 2018?", [build_document()]
        )
        assert (
            a.text == "1,577 million"
            and a.found
            and a.grounded
            and not a.abstained
        )
        assert (
            a.citations[0].verified
            and a.pages == [("annual", 7)]
            and a.rounds == 1
        )
        prompt = backend.requests[0].messages[-1].content
        assert (
            "[annual p.7]" in prompt and "Question: What were capital" in prompt
        )
        assert a.usage.total_tokens > 0 and a.foveation.tokens <= 3000

    def test_unverifiable_quote_triggers_a_second_round_with_more_pages(self):
        f, backend = foveator(
            [
                reply(
                    answer="2,000 million",
                    citations=[
                        {
                            "doc": "annual",
                            "page": 7,
                            "quote": "were 2,000 million dollars exactly",
                        }
                    ],
                    need_pages=[{"doc": "annual", "page": 12}],
                ),
                reply(
                    answer="1.50 per share",
                    citations=[
                        {
                            "doc": "annual",
                            "page": 12,
                            "quote": "dividend declared was 1.50 per share",
                        }
                    ],
                ),
            ],
            max_rounds=2,
        )
        a = f.ask("dividend per share capital expenditures", [build_document()])
        assert a.rounds == 2 and a.grounded and a.text == "1.50 per share"
        assert len(backend.requests) == 2
        retry_prompt = backend.requests[1].messages[-1].content
        assert (
            "NOTE:" in retry_prompt
            and "was NOT found on annual p.7" in retry_prompt
        )

    def test_not_found_abstains(self):
        f, _ = foveator([reply(found=False, answer="")], max_rounds=1)
        a = f.ask("What is the airspeed of a swallow?", [build_document()])
        assert (
            a.abstained
            and a.text == NOT_FOUND
            and not a.grounded
            and not a.found
        )

    def test_flag_vs_abstain_for_ungrounded(self):
        bad = reply(
            answer="maybe 5",
            citations=[
                {
                    "doc": "annual",
                    "page": 3,
                    "quote": "totally made up text here",
                }
            ],
        )
        flagged, _ = foveator([bad], max_rounds=1)
        a = flagged.ask("capital", [build_document()])
        assert a.text == "maybe 5" and not a.grounded and not a.abstained
        strict, _ = foveator([bad], max_rounds=1, ungrounded="abstain")
        b = strict.ask("capital", [build_document()])
        assert b.abstained and b.text == NOT_FOUND

    def test_non_json_reply_is_kept_as_ungrounded_text(self):
        f, _ = foveator(["The answer is 1,577 million."], max_rounds=1)
        a = f.ask("capital expenditures", [build_document()])
        assert a.text == "The answer is 1,577 million." and not a.grounded

    def test_hallucinated_citation_fields_are_ignored_safely(self):
        junk = reply(
            answer="x",
            citations=[
                "str",
                {"doc": "nope", "page": 1, "quote": "a b c d"},
                {"doc": "annual", "page": "x"},
                {"doc": "annual", "page": 999, "quote": "q"},
            ],
            need_pages=[{"doc": "annual", "page": "zz"}, "bad"],
        )
        f, _ = foveator([junk], max_rounds=1)
        a = f.ask("capital", [build_document()])
        assert not a.grounded and all(not c.verified for c in a.citations)

    def test_backend_failure_propagates(self):
        f, _ = foveator([errors.PermanentBackendError("down")])
        with pytest.raises(errors.PermanentBackendError):
            f.ask("capital expenditures", [build_document()])

    def test_prompt_fences_pages_as_untrusted(self):
        doc = build_document(3)
        f, backend = foveator([reply(found=False)], max_rounds=1)
        f.ask("anything", [doc])
        system = backend.requests[0].messages[0].content
        assert (
            "untrusted" in system
            and "<pages>" in backend.requests[0].messages[-1].content
        )


class TestPlanAndConfig:
    def test_plan_makes_no_model_call_and_reports_fit(self):
        f, backend = foveator([], budget=2000)
        plan = f.plan("dividend", [build_document(200)])
        assert backend.requests == []
        assert plan.document_tokens > 20_000 and plan.foveation.tokens <= 2000
        assert plan.prompt_tokens <= 2000 and plan.window == 128_000
        assert plan.fits_whole_documents  # 38k tokens fit a 128k window

    def test_index_is_reusable_across_questions(self):
        f, _ = foveator([], budget=2000)
        index = f.index([build_document()])
        a = f.plan("dividend per share", index)
        b = f.plan("headcount employees", index)

        def full_pages(plan):
            return {
                x.page for x in plan.foveation.pages if x.tier.value == "full"
            }

        assert 12 in full_pages(a) and 19 in full_pages(b)

    def test_auto_budget_uses_model_window(self):
        rt = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(), model="gpt-4o-mini", tokenizer=TOK
        )
        assert Foveator(rt).resolved_budget() == int(128_000 * 0.95) - 1_500
        unknown = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(), model="mystery"
        )
        with pytest.raises(errors.ConfigError):
            Foveator(unknown).resolved_budget()

    @pytest.mark.parametrize(
        "kw",
        [
            {"ungrounded": "x"},
            {"max_rounds": 0},
            {"k": 0},
            {"answer_tokens": 1},
            {"budget": 10},
        ],
    )
    def test_validation(self, kw):
        rt = runtime_lib.Runtime(backend=faults.ScriptedBackend())
        with pytest.raises(errors.ConfigError):
            Foveator(rt, **kw)
