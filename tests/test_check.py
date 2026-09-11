"""Tests for :mod:`ceng.check`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ceng.check import (
    parse_probability,
    ppa_check,
    validate_tree,
)


@dataclass
class FakeBackend:
    """Scripted backend that returns probability answers per call."""

    name: str = "fake"
    responses: list[str] = None
    calls: list[list[dict]] = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = []
        if self.calls is None:
            self.calls = []

    def complete(self, messages, model, **kw):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeBackend ran out of responses")
        return self.responses.pop(0)


@pytest.fixture()
def tmp_cache(tmp_path):
    return str(tmp_path / "cache")


# --- probability parsing ---


def testparse_probability_extracts_decimal():
    assert parse_probability("0.42") == 0.42


def testparse_probability_extracts_integer_in_range():
    assert parse_probability("answer: 0") == 0.0
    assert parse_probability("answer: 1") == 1.0


def testparse_probability_extracts_from_prose():
    assert parse_probability("I think the answer is around 0.73.") == 0.73


def testparse_probability_clamps_outside_range():
    assert parse_probability("1.5") == 1.0
    assert parse_probability("-0.2") == 0.0


def testparse_probability_raises_on_no_number():
    with pytest.raises(ValueError, match="could not parse"):
        parse_probability("no number here")


def testparse_probability_raises_on_none():
    with pytest.raises(ValueError, match="no content"):
        parse_probability(None)


# --- tree validation ---


def testvalidate_tree_accepts_leaf():
    parsed = validate_tree([{"description": "all"}])
    assert len(parsed) == 1
    assert parsed[0].description == "all"
    assert parsed[0].prior == 1.0
    assert parsed[0].children == []


def testvalidate_tree_rejects_empty_description():
    with pytest.raises(ValueError, match="description"):
        validate_tree([{"description": ""}])


def testvalidate_tree_rejects_missing_description():
    with pytest.raises(ValueError, match="description"):
        validate_tree([{}])


def testvalidate_tree_rejects_out_of_range_prior():
    with pytest.raises(ValueError, match="prior"):
        validate_tree([{"description": "x", "prior": 1.5}])


def testvalidate_tree_rejects_children_summing_to_more_than_one():
    with pytest.raises(ValueError, match="sum to"):
        validate_tree(
            [
                {
                    "description": "all",
                    "children": [
                        {"description": "a", "prior": 0.8},
                        {"description": "b", "prior": 0.8},
                    ],
                }
            ]
        )


def testvalidate_tree_accepts_nested():
    parsed = validate_tree(
        [
            {
                "description": "root",
                "children": [
                    {"description": "a", "prior": 0.5},
                    {"description": "b", "prior": 0.5},
                ],
            }
        ]
    )
    assert parsed[0].description == "root"
    assert len(parsed[0].children) == 2


# --- ppa_check core ---


def test_ppa_check_aggregates_leaves(tmp_cache):
    tree = [
        {
            "description": "EU",
            "prior": 0.3,
            "children": [
                {"description": "DE", "prior": 0.5},
                {"description": "FR", "prior": 0.5},
            ],
        },
        {"description": "US", "prior": 0.7},
    ]
    # Backend responses: population question, then DE, FR, US
    backend = FakeBackend(responses=["0.5", "0.4", "0.6", "0.5"])
    verdict = ppa_check(
        "prefer feature X?",
        "users",
        tree,
        llm="m",
        tolerance=0.01,
        backend=backend,
        cache_dir=tmp_cache,
    )
    # population = 0.5
    # aggregated = 0.3*0.5*0.4 + 0.3*0.5*0.6 + 0.7*0.5
    #            = 0.06 + 0.09 + 0.35 = 0.5
    assert verdict.population_estimate == 0.5
    assert abs(verdict.aggregated_estimate - 0.5) < 1e-9
    assert verdict.self_consistent is True
    assert verdict.delta < 0.01


def test_ppa_check_detects_macro_fallacy(tmp_cache):
    tree = [
        {"description": "A", "prior": 0.5},
        {"description": "B", "prior": 0.5},
    ]
    # population is 0.0 but each leaf thinks 1.0 (the macro fallacy)
    backend = FakeBackend(responses=["0.0", "1.0", "1.0"])
    verdict = ppa_check(
        "Q?",
        "pop",
        tree,
        llm="m",
        tolerance=0.1,
        backend=backend,
        cache_dir=tmp_cache,
    )
    assert verdict.population_estimate == 0.0
    assert verdict.aggregated_estimate == 1.0
    assert verdict.self_consistent is False
    assert verdict.delta == 1.0


def test_ppa_check_uses_cache_on_repeat(tmp_cache):
    tree = [{"description": "all"}]
    backend = FakeBackend(responses=["0.7", "0.7"])  # pop + leaf
    ppa_check("Q?", "pop", tree, llm="m", backend=backend, cache_dir=tmp_cache)
    assert len(backend.calls) == 2
    backend.calls = []
    backend.responses = []
    verdict = ppa_check(
        "Q?", "pop", tree, llm="m", backend=backend, cache_dir=tmp_cache
    )
    assert len(backend.calls) == 0
    assert verdict.cache_hits >= 1


def test_ppa_check_reports_per_leaf_breakdown(tmp_cache):
    tree = [
        {"description": "A", "prior": 0.4},
        {"description": "B", "prior": 0.6},
    ]
    backend = FakeBackend(responses=["0.5", "0.2", "0.8"])
    verdict = ppa_check(
        "Q?", "pop", tree, llm="m", tolerance=0.6, backend=backend, cache_dir=tmp_cache
    )
    descriptions = {leaf.description for leaf in verdict.leaves}
    assert descriptions == {"A", "B"}
    priors = {leaf.description: leaf.prior for leaf in verdict.leaves}
    assert priors == {"A": 0.4, "B": 0.6}


def test_ppa_check_aggregates_with_nested_priors(tmp_cache):
    tree = [
        {
            "description": "EU",
            "prior": 0.4,
            "children": [
                {"description": "DE", "prior": 0.7},  # effective 0.28
                {"description": "FR", "prior": 0.3},  # effective 0.12
            ],
        },
        {"description": "US", "prior": 0.6},  # effective 0.6
    ]
    backend = FakeBackend(
        responses=["0.0", "0.0", "0.0", "1.0"]  # pop, DE=0, FR=0, US=1
    )
    verdict = ppa_check(
        "Q?",
        "pop",
        tree,
        llm="m",
        backend=backend,
        cache_dir=tmp_cache,
    )
    assert abs(verdict.aggregated_estimate - 0.6) < 1e-9


def test_ppa_check_rejects_negative_tolerance(tmp_cache):
    with pytest.raises(ValueError, match="tolerance"):
        ppa_check(
            "Q?", "pop", [{"description": "all"}], llm="m",
            tolerance=-0.1, backend=FakeBackend(responses=[]),
            cache_dir=tmp_cache,
        )


def test_ppa_check_rejects_empty_tree(tmp_cache):
    with pytest.raises(ValueError, match="non-empty"):
        ppa_check("Q?", "pop", [], llm="m", backend=FakeBackend(),
                  cache_dir=tmp_cache)


def test_ppa_check_kw_forwarded(tmp_cache):
    backend = FakeBackend(responses=["0.5", "0.5"])
    ppa_check(
        "Q?", "pop", [{"description": "all"}], llm="m",
        backend=backend, cache_dir=tmp_cache, temperature=0.3,
    )
    assert True


def test_ppa_check_with_no_cache(tmp_cache):
    backend = FakeBackend(responses=["0.5", "0.5"])
    verdict = ppa_check(
        "Q?", "pop", [{"description": "all"}], llm="m",
        backend=backend, cache_dir="",
    )
    assert verdict.population_estimate == 0.5
    assert verdict.self_consistent is True
