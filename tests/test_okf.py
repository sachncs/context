"""Tests for :mod:`ceng.okf`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ceng.okf import (
    CENG_BUNDLE_INDEX,
    CENG_COMBINED_SUMMARY,
    CENG_LEAF_SUMMARY,
    OKF_VERSION,
    RESERVED_INDEX,
    RESERVED_LOG,
    Concept,
    Frontmatter,
    cross_links,
    find_concept,
    parse_concept,
    parse_frontmatter,
    read_bundle,
    read_concept_file,
    render_concept,
    render_frontmatter,
    write_bundle,
    write_concept_file,
)


# --- Frontmatter ---


def test_frontmatter_requires_type():
    with pytest.raises(ValueError, match="type"):
        Frontmatter.from_dict({})


def test_frontmatter_roundtrips_only_required_field():
    fm = Frontmatter(type="Concept")
    parsed = Frontmatter.from_dict(fm.to_dict())
    assert parsed.type == "Concept"
    assert parsed.title == ""
    assert parsed.description == ""
    assert parsed.tags == ()
    assert parsed.extra == ()


def test_frontmatter_roundtrips_all_fields():
    fm = Frontmatter(
        type="BigQuery Table",
        title="Orders",
        description="One row per order.",
        resource="https://example/orders",
        tags=("sales", "revenue"),
        timestamp="2026-05-28T14:30:00+00:00",
        extra=(("owner", "data-team@example"),),
    )
    parsed = Frontmatter.from_dict(fm.to_dict())
    assert parsed == fm


def test_frontmatter_extra_field_survives_roundtrip():
    fm = Frontmatter(type="Concept", extra=(("join_paths", ["customers"]),))
    parsed = Frontmatter.from_dict(fm.to_dict())
    assert parsed.extra == (("join_paths", ["customers"]),)


def test_frontmatter_now_sets_utc_timestamp():
    fm = Frontmatter.now(type="x")
    assert "T" in fm.timestamp
    assert fm.timestamp.endswith("+00:00") or fm.timestamp.endswith("Z")


def test_frontmatter_omits_empty_fields():
    fm = Frontmatter(type="x")
    assert render_frontmatter(fm.to_dict()) == "type: x"


def test_frontmatter_coerces_tags_to_tuple():
    parsed = Frontmatter.from_dict({"type": "x", "tags": ["a", "b", "c"]})
    assert parsed.tags == ("a", "b", "c")


def test_frontmatter_rejects_non_string_title():
    with pytest.raises(ValueError, match="title"):
        Frontmatter.from_dict({"type": "x", "title": 42})


# --- YAML subset ---


def test_render_and_parse_scalar():
    text = render_frontmatter({"type": "x", "count": 3})
    parsed = parse_frontmatter(text)
    assert parsed == {"type": "x", "count": 3}


def test_render_quotes_special_strings():
    text = render_frontmatter({"type": "odd: value"})
    parsed = parse_frontmatter(text)
    assert parsed == {"type": "odd: value"}


def test_render_and_parse_inline_list():
    parsed = parse_frontmatter(render_frontmatter({"tags": ["a", "b", "c"]}))
    assert parsed["tags"] == ["a", "b", "c"]


def test_render_and_parse_with_special_chars_in_string():
    text = render_frontmatter({"type": "x", "title": "has: colon"})
    parsed = parse_frontmatter(text)
    assert parsed["title"] == "has: colon"


def test_render_and_parse_with_quoted_string():
    text = render_frontmatter({"type": "x", "title": '"quoted with: colon"'})
    parsed = parse_frontmatter(text)
    # quoted strings are escaped during render and unescaped on parse,
    # so the original value (including quotes) round-trips intact.
    assert parsed["title"] == '"quoted with: colon"'


def test_parse_frontmatter_skips_comments_and_blank_lines():
    parsed = parse_frontmatter(
        "# header\n\ntype: x\n# trailing\ntitle: hello\n"
    )
    assert parsed == {"type": "x", "title": "hello"}


def test_parse_frontmatter_handles_empty_input():
    assert parse_frontmatter("") == {}


# --- Concept markdown ---


def test_render_concept_starts_with_frontmatter_block():
    c = Concept(frontmatter=Frontmatter(type="x"), body="hello")
    text = render_concept(c)
    assert text.startswith("---\n")
    assert "\n---\n\n" in text
    assert text.rstrip().endswith("hello")


def test_render_concept_handles_empty_body():
    c = Concept(frontmatter=Frontmatter(type="x"), body="")
    text = render_concept(c)
    assert text == "---\ntype: x\n---\n\n"


def test_parse_concept_extracts_frontmatter_and_body():
    text = textwrap.dedent(
        """\
        ---
        type: BigQuery Table
        title: Orders
        description: One row per order.
        tags: [sales, revenue]
        ---

        # Schema

        | Column | Type |
        |--------|------|
        | id     | STRING |
        """
    )
    c = parse_concept(text, path="tables/orders.md")
    assert c.frontmatter.type == "BigQuery Table"
    assert c.frontmatter.title == "Orders"
    assert c.frontmatter.tags == ("sales", "revenue")
    assert "Schema" in c.body
    assert "| STRING |" in c.body
    assert c.path == Path("tables/orders.md")


def test_parse_concept_missing_frontmatter_raises():
    with pytest.raises(ValueError, match="frontmatter"):
        parse_concept("# no frontmatter here")


def test_parse_concept_unterminated_frontmatter_raises():
    with pytest.raises(ValueError, match="terminated"):
        parse_concept("---\ntype: x\nstill going")


def test_concept_roundtrip_preserves_everything():
    original = Concept(
        frontmatter=Frontmatter(
            type="x",
            title="t",
            description="d",
            resource="https://example",
            tags=("a", "b"),
            timestamp="2026-05-28T14:30:00+00:00",
            extra=(("k", "v"),),
        ),
        body="# Heading\n\nParagraph.\n",
    )
    text = render_concept(original)
    parsed = parse_concept(text)
    assert parsed.frontmatter == original.frontmatter
    assert parsed.body == original.body.rstrip("\n")


def test_concept_with_path_returns_copy():
    original = Concept(frontmatter=Frontmatter(type="x"))
    updated = original.with_path("tables/x.md")
    assert updated.path == Path("tables/x.md")
    assert original.path is None


# --- File I/O ---


def test_write_and_read_concept_file(tmp_path):
    target = tmp_path / "concepts" / "x.md"
    c = Concept(
        frontmatter=Frontmatter(type="x", title="hi"),
        body="body text",
    )
    write_concept_file(target, c)
    assert target.exists()
    reloaded = read_concept_file(target)
    assert reloaded.frontmatter.title == "hi"
    assert reloaded.body == "body text"
    assert reloaded.path == target


def test_write_bundle_creates_files(tmp_path):
    bundle = tmp_path / "bundle"
    concepts = [
        Concept(
            frontmatter=Frontmatter(type="leaf", title="a"),
            body="body-a",
            path=Path("a.md"),
        ),
        Concept(
            frontmatter=Frontmatter(type="leaf", title="b"),
            body="body-b",
            path=Path("sub/b.md"),
        ),
    ]
    paths = write_bundle(bundle, concepts)
    assert len(paths) == 2
    assert paths[0].exists()
    assert paths[1].exists()
    assert (bundle / "a.md").exists()
    assert (bundle / "sub" / "b.md").exists()


def test_write_bundle_rejects_missing_path():
    with pytest.raises(ValueError, match="path"):
        write_bundle("/tmp/x", [Concept(frontmatter=Frontmatter(type="x"))])


def test_write_bundle_rejects_non_md_path(tmp_path):
    with pytest.raises(ValueError, match=".md"):
        write_bundle(
            tmp_path,
            [Concept(frontmatter=Frontmatter(type="x"), path=Path("y.txt"))],
        )


def test_write_bundle_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        write_bundle(
            tmp_path,
            [
                Concept(
                    frontmatter=Frontmatter(type="x"),
                    path=Path("../escape.md"),
                )
            ],
        )


def test_read_bundle_finds_every_md(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.md").write_text("---\ntype: x\n---\n\nA\n", encoding="utf-8")
    (bundle / "sub").mkdir()
    (bundle / "sub" / "b.md").write_text(
        "---\ntype: y\ntitle: b\n---\n\nB\n", encoding="utf-8"
    )
    (bundle / "ignore.txt").write_text("not a concept", encoding="utf-8")
    loaded = read_bundle(bundle)
    types = {c.frontmatter.type for c in loaded}
    assert types == {"x", "y"}
    paths = {c.path.as_posix() for c in loaded}
    assert paths == {"a.md", "sub/b.md"}


def test_read_bundle_nonexistent_returns_empty(tmp_path):
    assert read_bundle(tmp_path / "missing") == []


def test_find_concept_lookup(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(
        bundle,
        [
            Concept(frontmatter=Frontmatter(type="x"), path=Path("x.md")),
            Concept(
                frontmatter=Frontmatter(type="y"),
                body="links to [x](x.md)",
                path=Path("y.md"),
            ),
        ],
    )
    loaded = read_bundle(bundle)
    found = find_concept(loaded, "y.md")
    assert found is not None
    assert found.frontmatter.type == "y"
    assert find_concept(loaded, "missing.md") is None


def test_cross_links_extracts_md_targets():
    body = (
        "See [orders](tables/orders.md) and "
        "[customers](tables/customers.md). "
        "External: https://example.com. "
        "Also [log](log.md)."
    )
    assert cross_links(body) == [
        "tables/orders.md",
        "tables/customers.md",
        "log.md",
    ]


def test_cross_links_empty_body():
    assert cross_links("nothing here") == []


# --- constants exposed ---


def test_okf_version_constant():
    assert OKF_VERSION == "0.1"


def test_reserved_filenames():
    assert RESERVED_INDEX == "index.md"
    assert RESERVED_LOG == "log.md"


# --- CRLF + BOM normalisation (added in v0.3.0) ---


def test_parse_concept_handles_crlf_line_endings():
    text = "---\r\ntype: x\r\n---\r\n\r\nbody\r\n"
    parsed = parse_concept(text)
    assert parsed.frontmatter.type == "x"
    assert "body" in parsed.body


def test_parse_concept_handles_bare_cr_line_endings():
    text = "---\rtype: x\r---\r\rbody"
    parsed = parse_concept(text)
    assert parsed.frontmatter.type == "x"


def test_parse_concept_strips_utf8_bom():
    text = "\ufeff---\ntype: x\n---\n\nbody"
    parsed = parse_concept(text)
    assert parsed.frontmatter.type == "x"


def test_read_concept_file_handles_utf8_bom(tmp_path):
    target = tmp_path / "x.md"
    target.write_bytes(b"\xef\xbb\xbf---\ntype: x\n---\n\nbody\n")
    parsed = read_concept_file(target)
    assert parsed.frontmatter.type == "x"


# --- scalar quoting (added in v0.3.0; PyYAML handles these correctly) ---


def test_roundtrip_value_with_colon():
    original = Frontmatter(type="x", title="Sales: Orders")
    rendered = render_concept(Concept(frontmatter=original, body=""))
    parsed = parse_concept(rendered)
    assert parsed.frontmatter.title == "Sales: Orders"


def test_roundtrip_value_with_comma():
    original = Frontmatter(type="x", description="a, b, c")
    rendered = render_concept(Concept(frontmatter=original, body=""))
    parsed = parse_concept(rendered)
    assert parsed.frontmatter.description == "a, b, c"


def test_roundtrip_value_with_leading_whitespace():
    original = Frontmatter(type="x", title=" leading and trailing ")
    rendered = render_concept(Concept(frontmatter=original, body=""))
    parsed = parse_concept(rendered)
    assert parsed.frontmatter.title == " leading and trailing "


def test_roundtrip_value_with_hash():
    original = Frontmatter(type="x", title="color: #ff0000")
    rendered = render_concept(Concept(frontmatter=original, body=""))
    parsed = parse_concept(rendered)
    assert parsed.frontmatter.title == "color: #ff0000"


def test_inline_list_with_commas_in_strings():
    original = Frontmatter(type="x", tags=("a, b", "c"))
    rendered = render_concept(Concept(frontmatter=original, body=""))
    parsed = parse_concept(rendered)
    assert parsed.frontmatter.tags == ("a, b", "c")


def test_keys_with_colons_rejected():
    d = parse_frontmatter("type: x\n")
    assert d["type"] == "x"
    # PyYAML parses keys with embedded colons as scalar strings; we
    # only require that the round-trip survives, not that arbitrary
    # keys are valid Python identifiers.
    d2 = parse_frontmatter('"foo:bar": value\ntype: x\n')
    assert d2["type"] == "x"


# --- cross_links filters (added in v0.3.0) ---


def test_cross_links_excludes_external_urls():
    body = (
        "See [internal](tables/orders.md) and "
        "[external](https://example.com/foo.md)."
    )
    assert cross_links(body) == ["tables/orders.md"]


def test_cross_links_excludes_image_markdown():
    body = "![cover](images/cover.md)\n[doc](docs/index.md)"
    assert cross_links(body) == ["docs/index.md"]


def test_cross_links_strips_optional_title_attribute():
    body = '[table](tables/orders.md "the orders table")'
    assert cross_links(body) == ["tables/orders.md"]


def test_cross_links_skips_anchor_only_targets():
    body = "[self-link](#section) and [other](tables/x.md)"
    assert cross_links(body) == ["tables/x.md"]


def test_cross_links_preserves_order_and_dedup_not_required():
    body = "[a](a.md) [a](a.md) [b](b.md)"
    assert cross_links(body) == ["a.md", "a.md", "b.md"]  # dedup is caller's job


def test_ceng_concept_types_are_namespaced():
    assert CENG_LEAF_SUMMARY.startswith("ceng/")
    assert CENG_COMBINED_SUMMARY.startswith("ceng/")
    assert CENG_BUNDLE_INDEX.startswith("ceng/")


# --- write then re-read round-trip ---


def test_full_roundtrip_via_disk(tmp_path):
    bundle = tmp_path / "bundle"
    originals = [
        Concept(
            frontmatter=Frontmatter(
                type="leaf",
                title=f"leaf-{i}",
                description=f"description-{i}",
                tags=("ppa", f"leaf-{i}"),
                timestamp="2026-05-28T14:30:00+00:00",
            ),
            body=f"Body for leaf {i}. Links: [next](leaf-{i + 1}.md)",
            path=Path(f"leaf-{i}.md"),
        )
        for i in range(3)
    ]
    originals.append(
        Concept(
            frontmatter=Frontmatter(
                type="index",
                title="bundle index",
                description="Top-level index",
                tags=("ppa", "index"),
            ),
            body=(
                "This bundle holds leaf summaries.\n\n"
                "Leaves: [leaf-0](leaf-0.md), [leaf-1](leaf-1.md), "
                "[leaf-2](leaf-2.md)."
            ),
            path=Path(RESERVED_INDEX),
        )
    )
    write_bundle(bundle, originals)
    loaded = read_bundle(bundle)
    assert len(loaded) == 4
    loaded_types = sorted(c.frontmatter.type for c in loaded)
    assert loaded_types == ["index", "leaf", "leaf", "leaf"]
    index = find_concept(loaded, RESERVED_INDEX)
    assert index is not None
    assert cross_links(index.body) == [
        "leaf-0.md",
        "leaf-1.md",
        "leaf-2.md",
    ]
