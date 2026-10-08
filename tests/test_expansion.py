import asyncio
import json

import pytest

from foveate import Foveator, errors
from foveate import runtime as runtime_lib
from foveate.documents import chunking
from foveate.selection import base, expansion
from foveate.tokenizers import HeuristicTokenizer
from tests import faults
from tests.test_longdoc import make_doc

TOK = HeuristicTokenizer()


def runtime_with(reply):
    return runtime_lib.Runtime(
        backend=faults.ScriptedBackend(default=reply), tokenizer=TOK
    )


def run(coro):
    return asyncio.run(coro)


def test_rewrites_are_parsed_deduplicated_and_capped():
    reply = lambda r: json.dumps(  # noqa: E731
        {"queries": ["a b", "A B", "Q ONE", "", "q two", "q three"]}
    )
    out = run(expansion.rewrites(runtime_with(reply), "q one", 3))
    # the duplicate, the empty string and the original question are dropped
    assert out == ["a b", "q two", "q three"]


@pytest.mark.parametrize(
    "reply",
    [
        lambda r: "not json",
        lambda r: json.dumps({"queries": "nope"}),
        lambda r: json.dumps({}),
    ],
)
def test_bad_replies_give_no_rewrites(reply):
    assert run(expansion.rewrites(runtime_with(reply), "q", 2)) == []


def test_backend_failure_gives_no_rewrites():
    rt = runtime_lib.Runtime(
        backend=faults.ScriptedBackend([errors.PermanentBackendError("x")] * 5),
        tokenizer=TOK,
    )
    assert run(expansion.rewrites(rt, "q", 2)) == []
    assert (
        run(expansion.rewrites(runtime_lib.Runtime.without_llm(), "q", 2)) == []
    )


def hit(page, score=1.0):
    return base.Hit(chunking.Chunk("d", page, 0, "t", 1), score)


def test_fuse_rewards_agreement_between_queries():
    first = [hit(1), hit(2), hit(3)]
    second = [hit(3), hit(2), hit(9)]
    fused = expansion.fuse([first, second], k=4)
    order = [h.chunk.page for h in fused]
    assert order[0] in (2, 3) and set(order) == {1, 2, 3, 9}
    assert len(expansion.fuse([first], k=2)) == 2


def test_expansion_finds_a_page_the_question_wording_misses():
    doc = make_doc(
        pages=40,
        facts={
            22: "Purchases of property, plant and equipment totalled 1,577 million.",
        },
    )

    def reply(request):
        text = request.messages[-1].content
        if "search queries" in text:
            return json.dumps(
                {"queries": ["purchases of property plant and equipment"]}
            )
        return json.dumps({"found": False, "answer": "", "citations": []})

    question = "What was the capital outlay?"
    plain = Foveator(runtime_with(reply), budget=4000)
    expanded = Foveator(runtime_with(reply), budget=4000, expand=1)
    miss = plain.plan(question, [doc]).foveation
    hit_plan = expanded.plan(question, [doc]).foveation
    tier = lambda fov: next(p.tier.name for p in fov.pages if p.page == 22)  # noqa: E731
    assert tier(miss) not in ("FULL", "CONDENSED")
    assert tier(hit_plan) == "FULL"


def test_expand_is_validated():
    with pytest.raises(errors.ConfigError):
        Foveator(runtime_with(lambda r: ""), expand=9)
