import asyncio
import json
import re

import pytest

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.bench import needle
from foveate.bench.needle import cases, haystack, runner, sources
from foveate.cache import base as cache_base
from foveate.tokenizers import HeuristicTokenizer
from tests import faults

TOK = HeuristicTokenizer()
NEEDLE_SET = [
    {
        "id": "0401",
        "needle": "Actually, {CHAR} lives next to {1}.",
        "questions": {
            "onehop": "Which character has been to {2}?",
            "twohop": "Which character has been to {3}?",
        },
        "character_set": ["Yuki", "Stuart"],
        "tests": {
            "T1": {
                "input_args": ["the Semper Opera House", "Dresden", "Saxony"]
            }
        },
    }
]


class TestHaystack:
    def test_prose_is_deterministic_and_big_enough(self):
        a = haystack.prose_pages(4000, TOK, seed=1)
        assert a == haystack.prose_pages(4000, TOK, seed=1)
        assert a != haystack.prose_pages(4000, TOK, seed=2)
        assert sum(TOK.count(p) for p in a) >= 4000 and len(a) >= 7

    def test_prose_needs_at_least_a_page(self):
        with pytest.raises(errors.ConfigError):
            haystack.prose_pages(10, TOK, seed=0)

    def test_book_pages_start_at_a_seeded_offset(self):
        text = "\n\n".join(
            f"Paragraph {i}. " + "word " * 120 for i in range(40)
        )
        a = haystack.book_pages(text, 2500, TOK, seed=3)
        assert a == haystack.book_pages(text, 2500, TOK, seed=3)
        starts = {
            haystack.book_pages(text, 600, TOK, seed=n)[0][:14]
            for n in range(8)
        }
        assert len(starts) > 1
        assert sum(TOK.count(p) for p in a) >= 2500
        with pytest.raises(errors.ValidationError):
            haystack.book_pages("   ", 500, TOK, seed=0)

    @pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_insert_hides_the_sentence_at_the_depth(self, depth):
        pages = haystack.prose_pages(8000, TOK, seed=0)
        out = haystack.insert(pages, "NEEDLE SENTENCE.", depth)
        where = [
            i for i, p in enumerate(out, start=1) if "NEEDLE SENTENCE." in p
        ]
        assert where == [haystack.page_of(len(pages), depth)]
        assert out[where[0] - 1].count("\n") > pages[where[0] - 1].count("\n")
        assert not out[where[0] - 1].startswith("NEEDLE")

    def test_insert_validates_depth(self):
        with pytest.raises(errors.ConfigError):
            haystack.insert(["a"], "x", 1.5)

    def test_document_page_numbers_start_at_one(self):
        doc = haystack.to_document("h", ["a b", "c d"], TOK)
        assert doc.numbers == [1, 2] and doc.id == "h"


class TestCases:
    def test_literal_shares_words_with_its_question(self):
        case = cases.literal(8000, 0.5, seed=1)
        topic = case.question.removeprefix(
            "What is the internal codename of the "
        ).rstrip("?")
        assert topic in case.sentences[0] and case.expected in case.sentences[0]
        assert case == cases.literal(8000, 0.5, seed=1)
        assert case.expected != cases.literal(8000, 0.5, seed=2).expected

    def test_multi_spreads_needles_and_requires_all_values(self):
        case = cases.multi(16000, 3, seed=0, depth=0.0)
        assert len(case.sentences) == 3 and case.depths == (0.0, 1 / 3, 2 / 3)
        values = case.expected.split("|")
        assert case.correct(" and ".join(values)) and not case.correct(
            values[0]
        )
        late = cases.multi(16000, 2, seed=0, depth=0.5)
        assert late.depth == 0.5 and late.depths[1] > 0.5
        with pytest.raises(errors.ConfigError):
            cases.multi(8000, 1, seed=0)

    def test_nonliteral_question_shares_no_words_with_the_needle(self):
        case = cases.nonliteral(NEEDLE_SET, 0, "T1", 8000, 0.25, seed=0)
        assert case.expected in ("Yuki", "Stuart")
        assert "Semper" in case.sentences[0] and "Dresden" in case.question
        assert "Semper" not in case.question
        assert case.correct(f"It is {case.expected.lower()}.")
        two = cases.nonliteral(NEEDLE_SET, 0, "T1", 8000, 0.25, 0, hop="twohop")
        assert "Saxony" in two.question
        for bad in (dict(index=3), dict(test="nope"), dict(hop="threehop")):
            args = dict(index=0, test="T1", hop="onehop") | bad
            with pytest.raises(errors.ConfigError):
                cases.nonliteral(
                    NEEDLE_SET,
                    args["index"],
                    args["test"],
                    8000,
                    0.0,
                    0,
                    args["hop"],
                )

    def test_grid_covers_every_length_and_depth(self):
        grid = cases.grid(lengths=(8000, 16000), depths=(0.0, 1.0))
        assert len(grid) == 4 and {c.length for c in grid} == {8000, 16000}

    def test_case_validation(self):
        with pytest.raises(errors.ConfigError):
            cases.Case("i", "weird", 1, (0.0,), ("s",), "q", "e", 0)
        with pytest.raises(errors.ConfigError):
            cases.Case("i", cases.LITERAL, 1, (), (), "q", "e", 0)


def finder(request):
    """A rule-based 'model': returns the value of the needle the question names."""
    text = request.messages[-1].content
    question = text.rsplit("Question:", 1)[-1]
    for line in text.splitlines():
        match = re.search(r"codename of the ([a-z ]+) is ([A-Z0-9-]+)\.", line)
        if match and match.group(1) in question:
            return json.dumps(
                {"found": True, "answer": match.group(2), "citations": []}
            )
    return json.dumps({"found": False, "answer": "", "citations": []})


def make_runtime(window=None):
    return runtime_lib.Runtime(
        backend=faults.ScriptedBackend(default=finder),
        cache=cache_base.MemoryCache(),
        tokenizer=TOK,
        context_window=window,
    )


class TestRunner:
    def test_build_document_hides_every_needle(self):
        case = cases.multi(8000, 2, seed=0)
        doc = runner.build_document(case, sources.ProseSource(TOK))
        for sentence in case.sentences:
            assert sentence in doc.text()
        assert doc.token_count >= 8000

    def test_report_grids_and_the_overflow_baseline(self):
        grid = cases.grid(lengths=(8000,), depths=(0.0, 1.0))
        report = asyncio.run(
            needle.run(
                grid,
                sources.ProseSource(TOK),
                make_runtime(window=6000),
                needle.Config(
                    pipelines=("full-context", "naive-rag", "foveate"),
                    budget=4000,
                ),
            )
        )
        by = {(c.pipeline, c.depth): c for c in report.cells()}
        assert (
            by[("full-context", 0.0)].overflow == 1
            and by[("full-context", 0.0)].accuracy == 0.0
        )
        assert (
            by[("naive-rag", 1.0)].accuracy == 1.0
            and by[("foveate", 0.0)].accuracy == 1.0
        )
        assert report.by_length()[("foveate", "literal")] == {8000: 1.0}
        md = report.to_markdown()
        assert "### foveate, literal" in md and "| 8,000 |" in md
        summary = report.summary()
        assert summary["model"] and summary["cells"]

    def test_write_creates_both_files(self, tmp_path):
        report = asyncio.run(
            needle.run(
                [cases.literal(8000, 0.5, 0)],
                sources.ProseSource(TOK),
                make_runtime(),
                needle.Config(pipelines=("foveate",)),
            )
        )
        path = report.write(tmp_path / "out")
        assert path.read_text().startswith("# Needle")
        assert json.loads((tmp_path / "out" / "summary.json").read_text())[
            "cells"
        ]

    def test_config_validation(self):
        for kw in (
            {"budget": 10},
            {"concurrency": 0},
            {"pipelines": ("nope",)},
        ):
            with pytest.raises(errors.ConfigError):
                needle.Config(**kw)


class TestSources:
    def test_nolima_download_is_cached(self, tmp_path, monkeypatch):
        calls = []

        def fake_fetch(url):
            calls.append(url)
            if url.endswith("needle_set.json"):
                return json.dumps(NEEDLE_SET).encode()
            return b"A paragraph.\n\nAnother paragraph.\n\n" * 50

        monkeypatch.setattr(sources.dataset, "fetch", fake_fetch)
        needles, book = sources.load_nolima(TOK, tmp_path)
        again, _ = sources.load_nolima(TOK, tmp_path)
        assert needles == again and len(calls) == 2
        assert book.pages(1000, 0)

    def test_malformed_needle_set_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources.dataset, "fetch", lambda url: b'{"a": 1}')
        with pytest.raises(errors.ValidationError):
            sources.load_nolima(TOK, tmp_path)
