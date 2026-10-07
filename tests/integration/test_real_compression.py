"""Compression, verification and persistence with a real model."""

import pytest

from ceng import Context, Message, Role, errors
from ceng.cache import base as cache_base
from ceng.compression import Budget, Overflow
from tests.integration.conftest import fold

FACTS = {
    "budget": (
        "Project Zephyr has an approved budget of 4.2 million euros.",
        "4.2 million",
    ),
    "lead": (
        "The lead engineer of Project Zephyr is Marguerite Okonkwo.",
        "Okonkwo",
    ),
    "city": (
        "The Zephyr prototype is assembled in the city of Tromso.",
        "Tromso",
    ),
    "deadline": (
        "The final delivery deadline is the 14th of March 2027.",
        "March 2027",
    ),
    "codename": (
        "The internal codename for the sensor module is Heron-9.",
        "Heron-9",
    ),
}
FILLER = (
    "The committee reviewed routine logistics, parking arrangements, catering "
    "options, and the schedule of recurring meetings without reaching any "
    "decision on them. "
)


def make_document() -> str:
    parts = []
    for sentence, _needle in FACTS.values():
        parts.append(FILLER * 14)
        parts.append(sentence + " ")
    parts.append(FILLER * 14)
    return "".join(parts)


def recalled(text: str) -> int:
    folded = fold(text)
    return sum(1 for _, needle in FACTS.values() if fold(needle) in folded)


@pytest.fixture(scope="module")
def document_context(runtime):
    return Context(
        (
            Message(Role.SYSTEM, "You are a careful analyst."),
            Message(Role.USER, make_document()),
        ),
        runtime,
    )


def test_ppa_fits_budget_and_keeps_the_facts(document_context):
    assert document_context.token_count > 1500
    out = document_context.compress("ppa", budget=450, leaf_tokens=400)
    report = out.report
    assert out.token_count <= 450 and not report.truncated
    assert report.usage.total_tokens > 0 and report.original_tokens > 1500
    assert any(step.name.startswith("leaf") for step in report.steps)
    assert out.messages[0].content == "You are a careful analyst."
    assert recalled(out.messages[1].content) >= 3, out.messages[1].content


def test_second_run_is_served_from_cache(document_context):
    again = document_context.compress("ppa", budget=450, leaf_tokens=400)
    assert again.report.cache_hits == len(again.report.steps)


def test_hierarchical_reaches_a_tight_budget(document_context):
    out = document_context.compress(
        "hierarchical", budget=Budget(150, Overflow.TRUNCATE), leaf_tokens=400
    )
    assert out.token_count <= 150
    assert out.report.usage.total_tokens > 0


def test_ushape_summarises_the_middle_of_a_conversation(runtime):
    turns = []
    for i in range(12):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        turns.append(
            Message(
                role,
                f"Turn {i}: " + FILLER * 8 + f"Secret code {i} is {i * 111}.",
            )
        )
    ctx = Context(tuple(turns), runtime)
    out = ctx.compress("ushape", budget=1800, head=2, tail=3)
    assert (
        out.messages[:2] == ctx.messages[:2]
        and out.messages[-3:] == ctx.messages[-3:]
    )
    assert out.messages[2].content.startswith(
        "[Summary of earlier conversation]"
    )
    assert out.report.original_tokens > out.report.final_tokens > 0


def test_pipeline_uses_cheap_stage_first(runtime, document_context):
    out = document_context.compress(
        "truncate+ppa", budget=1200, ppa={"leaf_tokens": 400}
    )
    assert out.report.method == "truncate+ppa" and out.token_count <= 1200


def test_fallback_degrades_to_offline_when_the_provider_is_unreachable(
    document_context,
):
    import dataclasses

    from tests.integration import conftest

    broken = dataclasses.replace(
        document_context.runtime,
        backend=conftest.make_backend(api_key="nvapi-invalid"),
        cache=cache_base.NullCache(),
    )
    ctx = dataclasses.replace(document_context, runtime=broken)
    with pytest.raises(errors.CengError):
        ctx.compress("ppa", budget=450)
    out = ctx.compress("ppa|extractive", budget=450)
    assert out.token_count <= 450
    assert any(s.name.startswith("fallback after") for s in out.report.steps)


def test_llm_free_methods_need_no_provider():
    from ceng import Runtime

    ctx = Context((Message(Role.USER, make_document()),), Runtime.without_llm())
    for method in ("truncate", "extractive", "window+extractive"):
        out = ctx.compress(method, budget=300)
        assert out.token_count <= 300, method
    with pytest.raises(errors.CengError):
        ctx.compress("ppa", budget=300)


def test_macro_fallacy_probe_with_real_probabilities(runtime):
    ctx = Context((Message(Role.USER, "probe"),), runtime)
    verdict = ctx.verify(
        "macro_fallacy",
        question="What fraction of adults in the country own a bicycle?",
        population="adults in the Netherlands",
        tree=[
            {"description": "adults living in large cities", "prior": 0.4},
            {
                "description": "adults living in towns and villages",
                "prior": 0.6,
            },
        ],
        tolerance=0.5,
    )
    assert 0.0 <= verdict.population_estimate <= 1.0
    assert len(verdict.leaves) == 2
    assert all(0.0 <= leaf.estimate <= 1.0 for leaf in verdict.leaves)
    assert verdict.verifier == "macro_fallacy" and verdict.passed in (
        True,
        False,
    )


def test_real_report_survives_okf_and_json_roundtrips(
    tmp_path, document_context
):
    out = document_context.compress("ppa", budget=450, leaf_tokens=400)
    out.save(tmp_path / "bundle")
    out.save(tmp_path / "ctx.json", format="json")
    from_okf = Context.load(tmp_path / "bundle", runtime=out.runtime)
    from_json = Context.load(tmp_path / "ctx.json", "json", out.runtime)
    for loaded in (from_okf, from_json):
        assert loaded.messages == out.messages and loaded.report == out.report
