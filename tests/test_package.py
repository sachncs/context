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
    "configure_logging",
    "Verdict",
    "LeafEstimate",
    "ppa_check",
    "CompressError",
    "CompressionBundle",
    "LeafArtifact",
    "MAX_LEAVES",
    "compress_to_bundle",
    "ppa_compress",
    "ppa_compress_to_okf",
    "Concept",
    "Frontmatter",
    "now_iso",
    "parse_concept",
    "parse_frontmatter",
    "render_concept",
    "render_frontmatter",
    "read_bundle",
    "read_concept_file",
    "write_bundle",
    "write_concept_file",
    "cross_links",
    "find_concept",
    "OKF_VERSION",
    "RESERVED_INDEX",
    "RESERVED_LOG",
    "CENG_BUNDLE_INDEX",
    "CENG_COMBINED_SUMMARY",
    "CENG_LEAF_SUMMARY",
    "Partition",
    "partition_text",
    "count_tokens",
    "token_budget_split",
}


def test_version_is_semver_string():
    assert isinstance(ceng.__version__, str)
    # 0.4.0 is the current label: three numeric components
    # separated by dots, with an optional -suffix. The numeric
    # components must all be digits; the suffix is free-form.
    parts = ceng.__version__.split("-")[0].split(".")
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
    v = ceng.Verdict(
        question="q",
        population="pop",
        population_estimate=0.5,
        aggregated_estimate=0.4,
        self_consistent=True,
        delta=0.1,
        tolerance=0.2,
        leaves=(
            ceng.LeafEstimate(
                description="A", prior=1.0, estimate=0.4, cache_hit=False
            ),
        ),
        cache_hits=0,
        cache_misses=1,
    )
    assert v.self_consistent is True
    assert v.delta == 0.1


def test_default_backend_is_litellm_after_reset():
    ceng.reset_backend()
    backend = ceng.get_backend()
    assert backend.name == "litellm"


def test_concept_default_path_is_none():
    c = ceng.Concept(frontmatter=ceng.Frontmatter(type="x"))
    assert c.path is None
    assert c.body == ""


def test_now_iso_returns_string():
    out = ceng.now_iso()
    assert isinstance(out, str)
    assert "T" in out


def test_okf_version_constant_is_set():
    assert ceng.OKF_VERSION == "0.1"
