"""Smoke tests for the top-level :mod:`ceng` package surface."""

from __future__ import annotations

import ceng


EXPECTED_EXPORTS = {
    "Backend",
    "LiteLLMBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "available_backends",
    "get_backend",
    "reset_backend",
    "set_backend",
    "Cache",
    "make_key",
    "NAMESPACE_PPA_CHECK",
    "NAMESPACE_SUMMARIZE",
    "Verdict",
    "LeafEstimate",
    "ppa_check",
    "CompressResult",
    "compress_with_stats",
    "ppa_compress",
    "Partition",
    "partition_text",
    "count_tokens",
    "token_budget_split",
}


def test_version_is_string():
    assert isinstance(ceng.__version__, str)
    parts = ceng.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_all_exports_resolve():
    for name in EXPECTED_EXPORTS:
        assert hasattr(ceng, name), f"missing public export: {name}"


def test_all_attribute_matches_public_surface():
    assert set(ceng.__all__) == EXPECTED_EXPORTS


def test_namespace_constants_are_distinct_strings():
    assert isinstance(ceng.NAMESPACE_SUMMARIZE, str)
    assert isinstance(ceng.NAMESPACE_PPA_CHECK, str)
    assert ceng.NAMESPACE_SUMMARIZE != ceng.NAMESPACE_PPA_CHECK


def test_partition_is_dataclass_like():
    p = ceng.Partition(text="hello", index=0)
    assert p.text == "hello"
    assert p.index == 0


def test_verdict_is_dataclass_like():
    from ceng.check import LeafEstimate

    v = ceng.Verdict(
        question="q",
        population="pop",
        population_estimate=0.5,
        aggregated_estimate=0.4,
        self_consistent=True,
        delta=0.1,
        tolerance=0.2,
        leaves=(LeafEstimate(description="A", prior=1.0, estimate=0.4, cache_hit=False),),
        cache_hits=0,
        cache_misses=1,
    )
    assert v.self_consistent is True
    assert v.delta == 0.1


def test_default_backend_is_litellm_after_reset():
    ceng.reset_backend()
    backend = ceng.get_backend()
    assert backend.name == "litellm"
