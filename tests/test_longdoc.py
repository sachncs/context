import asyncio
import json

import pytest

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.bench.longdoc import dataset, gold, metrics, pipelines, runner
from foveate.cache import base as cache_base
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.tokenizers import HeuristicTokenizer
from tests import faults
from tests.test_documents import make_pdf

TOK = HeuristicTokenizer()


def run(coro):
    return asyncio.run(coro)


def make_doc(doc_id="acme_2018", pages=40, facts=None):
    facts = facts or {
        7: "Capital expenditures were 1,577 million dollars in fiscal 2018."
    }
    items = []
    for n in range(1, pages + 1):
        body = (
            facts.get(n, f"Routine discussion of operations on page {n}.")
            + " Filler. " * 60
        )
        items.append(page_lib.Page(n, body, TOK.count(body), (f"SECTION {n}",)))
    return Document(doc_id, tuple(items))


def question(**kw):
    base = dict(
        id="fb_1",
        company="Acme",
        doc_name="acme_2018",
        doc_link="http://x/a.pdf",
        kind="metrics-generated",
        question="What were Acme capital expenditures in fiscal 2018?",
        answer="$1577.00",
        justification="",
        evidence=(
            dataset.Evidence(
                "Capital expenditures were 1,577 million dollars in fiscal 2018.",
                6,
            ),
        ),
    )
    base.update(kw)
    return dataset.Question(**base)


class TestDataset:
    def test_parse_evidence_forms(self):
        raw = [
            {
                "evidence_text": "t",
                "evidence_page_num": 3,
                "evidence_text_full_page": "f",
            }
        ]
        assert dataset.parse_evidence(raw)[0].page_index == 3
        assert dataset.parse_evidence(str(raw))[0].text == "t"
        assert (
            dataset.parse_evidence("garbage(") == ()
            and dataset.parse_evidence(None) == ()
        )

    def test_loads_jsonl_and_pdf_from_the_cache_without_network(self, tmp_path):
        bench = dataset.FinanceBench(tmp_path)
        row = {
            "financebench_id": "fb_1",
            "company": "Acme",
            "doc_name": "ACME_2018_10K",
            "doc_link": "http://invalid.example/x.pdf",
            "question_type": "novel-generated",
            "question": "Q?",
            "answer": "A",
            "justification": "J",
            "evidence": [{"evidence_text": "e", "evidence_page_num": 0}],
        }
        (bench.root / "financebench_merged.jsonl").write_text(
            json.dumps(row) + "\n"
        )
        (bench.root / "pdf").mkdir()
        (bench.root / "pdf" / "ACME_2018_10K.pdf").write_bytes(make_pdf(3))
        q = bench.questions()[0]
        doc = bench.document(q, TOK)
        assert q.kind == "novel-generated" and len(doc.pages) == 3
        again = bench.document(q, TOK)  # served from the parsed cache
        assert [p.text for p in again.pages] == [p.text for p in doc.pages]

    def test_fetch_rejects_non_http_and_unreachable(self):
        with pytest.raises(errors.ValidationError):
            dataset.fetch("file:///etc/passwd")
        with pytest.raises(errors.ValidationError):
            dataset.fetch("http://127.0.0.1:9/none.pdf")

    def test_non_pdf_download_is_rejected(self, tmp_path):
        bench = dataset.FinanceBench(tmp_path)
        (tmp_path / "financebench" / "pdf").mkdir()
        # a non-PDF payload already cached must still fail to parse cleanly
        (tmp_path / "financebench" / "pdf" / "d.pdf").write_bytes(
            b"<html>nope</html>"
        )
        with pytest.raises(errors.ValidationError):
            bench.document(question(doc_name="d"), TOK)

    def test_stratified_sample_is_balanced_deterministic_and_wide(self):
        qs = [
            question(
                id=f"q{i}", doc_name=f"doc{i % 7}", kind=("a", "b", "c")[i % 3]
            )
            for i in range(60)
        ]
        s1 = dataset.stratified_sample(qs, 9, seed=1)
        assert s1 == dataset.stratified_sample(qs, 9, seed=1) and len(s1) == 9
        assert {q.kind for q in s1} == {"a", "b", "c"}
        assert len({q.doc_name for q in s1}) >= 5
        assert len(dataset.stratified_sample(qs, 500)) == 60


class TestGold:
    def test_locates_evidence_page_ignoring_the_dataset_index(self):
        doc = make_doc()
        found = gold.locate_pages(
            doc, question().evidence[0].text, hint_index=6
        )
        assert found[0] == 7
        assert (
            gold.locate_pages(doc, question().evidence[0].text, hint_index=30)[
                0
            ]
            == 7
        )
        assert gold.locate_pages(doc, "no such passage anywhere in here") == ()
        assert gold.locate_pages(doc, "two words") == ()

    def test_answerable_item(self):
        item = gold.answerable_item(question(), make_doc())
        assert item.kind == "answerable" and item.gold_pages == (7,)
        assert item.meta["number_on_page"] == "1"
        orphan = question(
            evidence=(dataset.Evidence("totally absent evidence text here", 0),)
        )
        assert gold.answerable_item(orphan, make_doc()) is None

    def test_unanswerable_requires_a_filing_that_lacks_the_answer(self):
        other = make_doc(
            "globex_2018", facts={5: "Globex revenue was 800 million."}
        )
        item = gold.unanswerable_item(question(), other)
        assert (
            item.kind == "unanswerable"
            and item.expected == ""
            and item.doc_name == "globex_2018"
        )
        assert (
            gold.unanswerable_item(question(), make_doc()) is None
        )  # same filing
        leaky = make_doc("leaky", facts={3: "Acme did report 1,577 elsewhere."})
        assert gold.unanswerable_item(question(), leaky) is None

    def test_needles_are_inserted_at_the_requested_depth(self):
        doc = make_doc(pages=20)
        items = gold.needle_items(doc, seed=3)
        assert [i.meta["depth"] for i in items] == [
            "0.0",
            "0.25",
            "0.5",
            "0.75",
            "1.0",
        ]
        assert (
            items[2].gold_pages[0] == 11
            and items[0].gold_pages == (1,)
            and items[4].gold_pages == (20,)
        )
        changed = gold.with_needle(doc, items[2])
        assert items[2].meta["sentence"] in changed.page(11).text
        assert items[2].meta["sentence"] not in doc.page(11).text
        assert items[2].expected in items[2].meta["sentence"]
        assert gold.needle_items(doc, seed=3) == items  # deterministic

    def test_jsonl_roundtrip_and_validation(self, tmp_path):
        items = gold.needle_items(make_doc(), seed=1)
        gold.write_items(tmp_path / "g.jsonl", items)
        assert gold.read_items(tmp_path / "g.jsonl") == sorted(
            items, key=lambda i: i.to_json()
        )
        with pytest.raises(errors.ValidationError):
            gold.GoldItem.from_json("{bad")
        with pytest.raises(errors.ValidationError):
            gold.GoldItem("i", "nope", "d", "q", "e")

    def test_key_numbers(self):
        assert gold.key_numbers("$1,577.00 and 7 and 24.8%") == [
            "1577.00",
            "24.8",
        ]


class TestMetrics:
    @pytest.mark.parametrize(
        "expected,predicted,verdict",
        [
            ("$1577.00", "1,577 million USD", True),
            ("$1577.00", "It was $1.577 billion", True),
            ("$1577.00", "about 1,600", False),
            ("24.8%", "24.8 percent", True),
            ("-$1.88", "(1.88) loss", True),
            ("0.66", "0.659", True),
            ("In FY2018 it was 8.74", "The EPS was 8.74 in 2018", True),
            ("$8.74", "$9.74", False),
            ("Yes", "yes", None),
            (
                "$1577",
                "I could not find this in the provided documents.",
                False,
            ),
        ],
    )
    def test_numeric_match(self, expected, predicted, verdict):
        assert metrics.numeric_match(expected, predicted) is verdict

    def test_scale_match_needs_a_scale_word(self):
        assert metrics.numeric_match("1577", "1.577") is False
        assert metrics.numeric_match("1577", "1.577 billion") is True

    def test_page_overlap_and_agreement(self):
        assert metrics.page_overlap([1, 2], [2, 3]) == (0.5, 0.5)
        assert metrics.page_overlap([], [1]) == (0.0, 0.0)
        item = lambda src, ok: metrics.Outcome(  # noqa: E731
            gold.GoldItem(
                f"i{src}{ok}", "paraphrase", "d", "q", "e", meta={"source": src}
            ),
            "p",
            0,
            pipelines.Result(),
            ok,
            "numeric",
        )
        outcomes = [
            item("a", True),
            item("a", True),
            item("b", True),
            item("b", False),
            item("c", True),
        ]
        assert metrics.agreement(outcomes) == 0.5  # c has a single outcome

    def test_score_paths(self):
        rt = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(
                default=lambda r: json.dumps({"correct": True})
            ),
            cache=cache_base.MemoryCache(),
        )
        item = gold.GoldItem("a:1", "answerable", "d", "Q", "1577", (7,))
        ok = pipelines.Result(text="1,577", cited_pages=(7,))
        out = run(metrics.score(rt, item, "p", 0, ok))
        assert out.correct and out.method == "numeric" and out.recall == 1.0
        text_item = gold.GoldItem(
            "a:2", "answerable", "d", "Q", "Yes, they did", (1,)
        )
        judged = run(
            metrics.score(
                rt, text_item, "p", 0, pipelines.Result(text="They did.")
            )
        )
        assert judged.correct and judged.method == "judge"
        failed = run(
            metrics.score(rt, item, "p", 0, pipelines.Result(error="boom"))
        )
        assert not failed.correct and failed.method == "error"
        abstained = run(
            metrics.score(rt, item, "p", 0, pipelines.Result(abstained=True))
        )
        assert not abstained.correct
        un = gold.GoldItem("u:1", "unanswerable", "d", "Q", "")
        assert run(
            metrics.score(rt, un, "p", 0, pipelines.Result(abstained=True))
        ).correct
        assert not run(
            metrics.score(rt, un, "p", 0, pipelines.Result(text="x"))
        ).correct
        needle = gold.GoldItem("n:1", "needle", "d", "Q", "Kalo-123", (3,))
        assert run(
            metrics.score(
                rt, needle, "p", 0, pipelines.Result(text="It is kalo-123.")
            )
        ).correct

    def test_judge_failure_counts_as_incorrect(self):
        rt = runtime_lib.Runtime(
            backend=faults.ScriptedBackend([errors.PermanentBackendError("x")])
        )
        assert run(metrics.judge(rt, "Q", "A", "B")) is False


def responder(request):
    """A rule-based model for harness tests: quotes the page holding the fact."""
    text = request.messages[-1].content
    if "Grade a candidate" in text:
        return json.dumps({"correct": True})
    if "capital expenditures" in text.lower() and "1,577" in text:
        return json.dumps(
            {
                "found": True,
                "answer": "1,577 million",
                "citations": [
                    {
                        "doc": "acme_2018",
                        "page": 7,
                        "quote": "Capital expenditures were 1,577 million dollars",
                    }
                ],
            }
        )
    return json.dumps({"found": False, "answer": "", "citations": []})


def rt(**kw):
    return runtime_lib.Runtime(
        backend=faults.ScriptedBackend(default=responder),
        cache=cache_base.MemoryCache(),
        tokenizer=TOK,
        **kw,
    )


class TestPipelines:
    q = "What were Acme capital expenditures in fiscal 2018?"

    def test_full_context_overflows_a_small_window(self):
        r = run(
            pipelines.FullContext(rt(context_window=5000), 3000).answer(
                self.q, make_doc()
            )
        )
        assert r.overflow and r.error and r.prompt_tokens == 0

    def test_full_context_answers_when_it_fits(self):
        r = run(
            pipelines.FullContext(rt(context_window=500_000), 3000).answer(
                self.q, make_doc()
            )
        )
        assert (
            r.found and r.cited_pages == (7,) and r.grounded and r.verified == 1
        )

    def test_truncate_keeps_only_the_first_pages(self):
        miss = run(
            pipelines.NaiveTruncate(rt(), 2000).answer(
                self.q, make_doc(pages=40)
            )
        )
        assert miss.abstained  # page 7 is past the truncation point
        hit = run(
            pipelines.NaiveTruncate(rt(), 12000).answer(
                self.q, make_doc(pages=40)
            )
        )
        assert hit.found and hit.cited_pages == (7,)

    def test_naive_rag_and_foveate_find_the_page(self):
        for cls in (pipelines.NaiveRag, pipelines.Foveate):
            r = run(cls(rt(), 4000).answer(self.q, make_doc(pages=80)))
            assert r.found and r.cited_pages == (7,), cls.__name__
            assert r.prompt_tokens > 0 and r.seconds >= 0

    def test_backend_errors_become_error_results(self):
        bad = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(
                [errors.PermanentBackendError("x")] * 3
            ),
            tokenizer=TOK,
        )
        for cls in (pipelines.NaiveRag, pipelines.Foveate):
            assert run(cls(bad, 4000).answer(self.q, make_doc())).error

    def test_sampled_runs_use_independent_tags(self):
        p = pipelines.NaiveRag(rt(), 4000)
        assert (
            p.foveator("r1").temperature == 0.7
            and p.foveator("").temperature == 0.0
        )


class TestRunnerAndReport:
    def corpus(self, tmp_path):
        bench = dataset.FinanceBench(tmp_path)
        c = runner.Corpus(bench, [question()], rt())
        c.cache["acme_2018"] = make_doc(pages=60)
        return c

    def test_end_to_end_report(self, tmp_path):
        doc = make_doc(pages=60)
        items = [
            gold.answerable_item(question(), doc),
            gold.GoldItem(
                "para:1",
                "paraphrase",
                "acme_2018",
                "How much capital expenditures did Acme report for fiscal 2018?",
                "$1577.00",
                (7,),
                {"source": "fb_1"},
            ),
            gold.GoldItem(
                "un:1",
                "unanswerable",
                "acme_2018",
                "What was Zorp's revenue?",
                "",
            ),
            gold.needle_items(doc, 1)[0],
        ]
        report = run(
            runner.run(
                items,
                self.corpus(tmp_path),
                rt(context_window=500_000),
                rt(),
                runner.Config(
                    pipelines=("full-context", "naive-rag", "foveate"),
                    budget=4000,
                    runs=2,
                    concurrency=2,
                ),
            )
        )
        rows = {r.pipeline: r for r in report.rows()}
        assert set(rows) == {"full-context", "naive-rag", "foveate"}
        assert (
            rows["foveate"].answerable == 1 and rows["foveate"].accuracy == 1.0
        )
        assert (
            rows["foveate"].grounded == 1.0 and rows["foveate"].page_hit == 1.0
        )
        assert (
            rows["foveate"].correct_abstain == 1.0
            and rows["foveate"].hallucinated == 0.0
        )
        md = report.to_markdown()
        for heading in (
            "Answer quality",
            "Grounding",
            "Reliability",
            "Cost and speed",
        ):
            assert heading in md
        path = report.write(tmp_path / "out")
        data = json.loads((tmp_path / "out" / "longdoc.json").read_text())
        assert path.exists() and len(data["outcomes"]) == len(report.outcomes)
        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert {r["pipeline"] for r in summary["rows"]} == set(rows)
        loaded = runner.Report.load(tmp_path / "out" / "longdoc.json", items)
        assert loaded.rows() == report.rows()
        with pytest.raises(errors.ValidationError):
            runner.Report.load(tmp_path / "out" / "longdoc.json", items[:1])

    def test_document_failure_is_scored_as_error_not_crash(self, tmp_path):
        c = runner.Corpus(
            dataset.FinanceBench(tmp_path),
            [question(doc_link="http://127.0.0.1:9/x.pdf")],
            rt(),
        )
        item = gold.GoldItem("a:1", "answerable", "acme_2018", "Q", "1", (1,))
        report = run(
            runner.run(
                [item], c, rt(), rt(), runner.Config(pipelines=("foveate",))
            )
        )
        assert report.rows()[0].errors == 1.0

    def test_config_validation(self):
        for kw in (
            {"runs": 0},
            {"concurrency": 0},
            {"budget": 10},
            {"pipelines": ("nope",)},
        ):
            with pytest.raises(errors.ConfigError):
                runner.Config(**kw)
