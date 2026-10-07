import os
import pathlib

import pytest

from ceng import Context, Message, Role, errors
from ceng import persistence
from ceng.compression import report as report_lib
from ceng.okf import Bundle, Concept, Frontmatter
from ceng.okf import codec as okf_codec
from ceng.okf import model
from tests.test_compression import ctx_of, make_runtime


def concept(path="a.md", **kw):
    return Concept(Frontmatter(type="t", **kw), "body text").at(path)


class TestFrontmatter:
    def test_roundtrip_all_fields(self):
        fm = Frontmatter(
            type="x",
            title="T",
            description="D",
            resource="R",
            tags=("a", "b"),
            timestamp="2025-01-01",
            priority=3,
            expires_at="2030",
            helpful_count=1,
            harmful_count=2,
            extra={"k": [1, "s"]},
        )
        assert Frontmatter.from_mapping(fm.to_mapping()) == fm

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"type": 3},
            {"type": "x", "priority": "hi"},
            {"type": "x", "priority": True},
            {"type": "x", "tags": "a"},
            {"type": "x", "tags": [1]},
            {"type": "x", "title": 1},
            {"type": "x", "nested": {"a": 1}},
        ],
    )
    def test_invalid(self, bad):
        with pytest.raises(errors.ValidationError):
            Frontmatter.from_mapping(bad)

    def test_requires_type_and_known_extra_keys(self):
        with pytest.raises(errors.ValidationError):
            Frontmatter(type="")
        with pytest.raises(errors.ValidationError):
            Frontmatter(type="x", extra={"title": "dup"})

    def test_scalars(self):
        assert model.is_scalar([1, ["a", None]])
        assert not model.is_scalar({"a": 1})


class TestConcept:
    def test_render_parse_roundtrip(self):
        c = concept(tags=("x",))
        parsed = Concept.parse(c.render(), c.path)
        assert parsed == c

    def test_crlf_bom_and_eof_terminator(self):
        text = "﻿---\r\ntype: x\r\n---\r\n\r\nhello\r\n"
        assert Concept.parse(text).body == "hello"
        assert Concept.parse("---\ntype: x\n---").body == ""

    @pytest.mark.parametrize(
        "text",
        [
            "no frontmatter",
            "---\ntype: x\nnever closed",
            "---\n- a\n- b\n---\n",
            "---\n: : :\n---\n",
        ],
    )
    def test_invalid(self, text):
        with pytest.raises(errors.ValidationError):
            Concept.parse(text)

    def test_empty_frontmatter_needs_type(self):
        with pytest.raises(errors.ValidationError):
            Concept.parse("---\n\n---\n")  # empty mapping -> no type

    def test_links(self):
        c = Concept(
            Frontmatter(type="x"),
            "[a](one.md) ![i](img.md) [e](https://x.org/a.md) [b](d/two.md)",
        )
        assert c.links() == ["one.md", "d/two.md"]

    def test_now_iso(self):
        assert "T" in model.now_iso()


class TestBundle:
    def test_write_read_queries(self, tmp_path):
        bundle = Bundle.of(
            [
                concept("a.md", tags=("x",), priority=1),
                concept("d/b.md", priority=5, helpful_count=2),
            ]
        )
        written = bundle.write(tmp_path / "out")
        assert len(written) == 2 and all(p.exists() for p in written)
        read = Bundle.read(tmp_path / "out")
        assert len(read) == 2 and list(read) == list(read.concepts)
        assert read.find("d/b.md").frontmatter.priority == 5
        assert read.find("zzz.md") is None
        assert [c.path.as_posix() for c in read.with_tag("x")] == ["a.md"]
        assert len(read.of_type("t")) == 2
        assert [c.path.as_posix() for c in read.with_priority_at_least(1)] == [
            "d/b.md",
            "a.md",
        ]

    def test_overwrite_is_atomic_and_clean(self, tmp_path):
        target = tmp_path / "out"
        Bundle.of([concept("old.md")]).write(target)
        Bundle.of([concept("new.md")]).write(target)
        assert sorted(p.name for p in target.iterdir()) == ["new.md"]
        assert [p.name for p in tmp_path.iterdir()] == ["out"]

    def test_failed_write_keeps_old_bundle(self, tmp_path, monkeypatch):
        target = tmp_path / "out"
        Bundle.of([concept("old.md")]).write(target)
        real = pathlib.Path.replace
        calls = {"n": 0}

        def flaky(self, other):
            calls["n"] += 1
            if calls["n"] == 2:  # the staging -> root swap
                raise OSError("disk full")
            return real(self, other)

        monkeypatch.setattr(pathlib.Path, "replace", flaky)
        with pytest.raises(OSError):
            Bundle.of([concept("new.md")]).write(target)
        monkeypatch.undo()
        assert [p.name for p in target.iterdir()] == ["old.md"]
        assert [p.name for p in tmp_path.iterdir()] == ["out"]

    @pytest.mark.parametrize(
        "path", ["/abs.md", "../escape.md", "no_suffix", "a/../../b.md"]
    )
    def test_bad_paths(self, path):
        with pytest.raises(errors.ValidationError):
            Bundle.of([concept(path)])

    def test_duplicate_and_unplaced(self):
        with pytest.raises(errors.ValidationError):
            Bundle.of([concept("a.md"), concept("a.md")])
        with pytest.raises(errors.ValidationError):
            Bundle.of([Concept(Frontmatter(type="t"))])

    def test_read_errors(self, tmp_path):
        with pytest.raises(errors.ValidationError):
            Bundle.read(tmp_path / "missing")
        (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(errors.ValidationError):
            Bundle.read(tmp_path)

    def test_write_onto_file_rejected(self, tmp_path):
        f = tmp_path / "f"
        f.write_text("x")
        with pytest.raises(errors.ValidationError):
            Bundle.of([]).write(f)

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges")
    def test_symlink_escape_refused(self, tmp_path):
        outside = tmp_path / "secret.md"
        outside.write_text("---\ntype: x\n---\nsecret")
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "link.md").symlink_to(outside)
        with pytest.raises(errors.ValidationError, match="escapes"):
            Bundle.read(bundle)


class TestCodecs:
    def test_okf_roundtrip_with_report_and_metadata(self, tmp_path):
        rt, _ = make_runtime()
        out = ctx_of(rt).compress("ppa", budget=300, leaf_tokens=128)
        out = Context(
            (*out.messages, Message(Role.ASSISTANT, "x", name="bot")),
            rt,
            out.report,
            {"run": "7"},
        )
        out.save(tmp_path / "ctx")
        loaded = Context.load(tmp_path / "ctx", runtime=rt)
        assert loaded.messages == out.messages
        assert loaded.report == out.report
        assert dict(loaded.metadata) == {"run": "7"}

    def test_okf_without_report(self, tmp_path):
        rt, _ = make_runtime()
        c = Context((Message(Role.USER, "hi"),), rt)
        c.save(tmp_path / "c")
        assert Context.load(tmp_path / "c", runtime=rt).report is None

    def test_json_roundtrip(self, tmp_path):
        rt, _ = make_runtime()
        out = ctx_of(rt).compress("truncate", budget=100)
        out.save(tmp_path / "c.json", format="json")
        loaded = Context.load(tmp_path / "c.json", "json", rt)
        assert loaded.messages == out.messages and loaded.report == out.report

    def test_not_a_context_bundle(self, tmp_path):
        Bundle.of([concept("a.md")]).write(tmp_path / "b")
        with pytest.raises(errors.ValidationError):
            okf_codec.OkfCodec().read(tmp_path / "b")

    def test_tampered_bundle_detected(self, tmp_path):
        rt, _ = make_runtime()
        Context((Message(Role.USER, "a"), Message(Role.USER, "b")), rt).save(
            tmp_path / "c"
        )
        (tmp_path / "c" / "messages" / "0001-user.md").unlink()
        with pytest.raises(errors.ValidationError, match="lists 2"):
            Context.load(tmp_path / "c", runtime=rt)

    def test_bad_role_and_report(self, tmp_path):
        rt, _ = make_runtime()
        Context((Message(Role.USER, "a"),), rt).save(tmp_path / "c")
        message = tmp_path / "c" / "messages" / "0000-user.md"
        message.write_text(
            message.read_text().replace("role: user", "role: bad")
        )
        with pytest.raises(errors.ValidationError, match="role"):
            Context.load(tmp_path / "c", runtime=rt)
        with pytest.raises(errors.ValidationError):
            okf_codec.OkfCodec.extract_json("no block")
        with pytest.raises(errors.ValidationError):
            okf_codec.OkfCodec.extract_json("```json\n{bad\n```")
        with pytest.raises(errors.ValidationError):
            okf_codec.OkfCodec.extract_json("```json\n[1]\n```")

    def test_json_errors_and_unknown_format(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{nope")
        with pytest.raises(errors.ValidationError):
            persistence.JsonCodec().read(bad)
        rt, _ = make_runtime()
        with pytest.raises(errors.ConfigError):
            Context((), rt).save(tmp_path / "x", format="yaml")

    def test_report_mapping_validation(self):
        with pytest.raises(errors.ValidationError):
            report_lib.CompressionReport.from_mapping({"method": "x"})
        with pytest.raises(errors.ValidationError):
            report_lib.StepRecord.from_mapping({})
