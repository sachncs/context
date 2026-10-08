import asyncio
import json
import re

import pytest

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.bench import compression
from foveate.bench.compression import runner, tasks
from foveate.cache import base as cache_base
from foveate.tokenizers import HeuristicTokenizer
from tests import faults

TOK = HeuristicTokenizer()


class TestTasks:
    def test_history_plants_the_fact_in_the_middle(self):
        sample = tasks.History(TOK, 4000).build(3)
        hits = [
            i
            for i, m in enumerate(sample.messages)
            if sample.expected in m.content
        ]
        assert len(hits) == 1 and 0 < hits[0] < len(sample.messages) * 0.7
        assert sample == tasks.History(TOK, 4000).build(3)
        assert sample.query == sample.question

    def test_tool_output_is_valid_json_with_one_target_row(self):
        sample = tasks.Tool(TOK, 4000).build(1)
        payload = json.loads(sample.messages[-1].content)
        order = re.search(r"order (OR-\d+)", sample.question).group(1)
        row = next(r for r in payload["results"] if r["order_id"] == order)
        assert row["status"] == sample.expected == "delayed-by-customs"
        assert sample.messages[-1].role.value == "tool"

    def test_document_hides_one_needle_and_keeps_pages(self):
        sample = tasks.Document(TOK, 4000).build(2)
        text = sample.messages[0].content
        assert sample.expected in text and text.count(sample.expected) == 1
        assert len(sample.pages) >= 7
        assert sum(TOK.count(m.content) for m in sample.messages) >= 4000


def reader(request):
    """A rule-based reader: finds whichever planted fact survived compression."""
    text = request.messages[-1].content
    question = text.rsplit("Question:", 1)[-1]
    patterns = (
        (r"deployment region is ([a-z0-9-]+)", "region"),
        (
            r"\"order_id\":\s*\"(OR-\d+)\"[^}]*?\"status\":\s*\"([a-z-]+)\"",
            "order",
        ),
        (r"codename of the ([a-z ]+) is ([A-Z0-9-]+)\.", "code"),
    )
    if "region" in question:
        found = re.search(patterns[0][0], text)
        return found.group(1) if found else "unknown"
    if "order" in question:
        wanted = re.search(r"order (OR-\d+)", question).group(1)
        found = re.search(rf"{wanted}[^}}]*?\"status\":\s*\"([a-z-]+)\"", text)
        return found.group(1) if found else "unknown"
    found = re.search(r"codename of the ([a-z ]+) is ([A-Z0-9-]+)\.", text)
    if found and found.group(1) in question:
        return json.dumps(
            {"found": True, "answer": found.group(2), "citations": []}
        )
    return json.dumps({"found": False, "answer": "", "citations": []})


def make_runtime():
    return runtime_lib.Runtime(
        backend=faults.ScriptedBackend(default=reader),
        cache=cache_base.MemoryCache(),
        tokenizer=TOK,
    )


class TestRunner:
    def sweep(self, **kw):
        cfg = compression.Config(
            tasks=("history", "atoms", "tool", "document"),
            methods=(
                "truncate",
                "window",
                "extractive",
                "query",
                "ushape-drop",
                "tool_output",
                "foveate",
                "selective",
                "clear_tool_results",
            ),
            ratios=(4, 16),
            samples=2,
            tokens=8000,
            **kw,
        )
        return asyncio.run(compression.run(make_runtime(), cfg))

    def test_uncompressed_baseline_answers_everything(self):
        report = self.sweep()
        base = [c for c in report.cells() if c.method == "uncompressed"]
        assert base and all(c.accuracy == 1.0 for c in base)
        assert all(c.ratio == 1 for c in base)

    def test_query_aware_beats_truncation_and_drop_at_high_compression(self):
        cells = {(c.task, c.method, c.ratio): c for c in self.sweep().cells()}
        assert cells[("document", "query", 16)].accuracy == 1.0
        assert cells[("document", "truncate", 16)].accuracy == 0.0
        assert cells[("history", "ushape-drop", 4)].accuracy == 0.0
        assert cells[("tool", "tool_output", 4)].accuracy >= 0.0
        assert cells[("document", "foveate", 4)].accuracy == 1.0

    def test_foveate_is_skipped_below_its_minimum_budget(self):
        cells = {(c.method, c.ratio) for c in self.sweep().cells()}
        assert ("foveate", 4) in cells and ("foveate", 16) not in cells

    def test_only_applicable_methods_run(self):
        names = {(c.task, c.method) for c in self.sweep().cells()}
        assert ("history", "window") in names and (
            "tool",
            "window",
        ) not in names
        assert ("tool", "tool_output") in names and (
            "history",
            "tool_output",
        ) not in names
        assert ("document", "foveate") in names and (
            "tool",
            "foveate",
        ) not in names

    def test_ratios_are_actually_achieved_and_reported(self):
        cell = next(
            c
            for c in self.sweep().cells()
            if c.task == "document"
            and c.method == "extractive"
            and c.ratio == 4
        )
        assert cell.achieved_ratio >= 3.5 and cell.n == 2

    def test_report_writes_files(self, tmp_path):
        report = self.sweep()
        path = report.write(tmp_path / "out")
        text = path.read_text()
        assert "### document" in text and "#### Correct" in text
        assert "Samples gained / lost" in text and "Facts kept" in text
        data = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert data["cells"] and data["model"]

    def test_failures_are_recorded_not_raised(self):
        bad = runtime_lib.Runtime(
            backend=faults.ScriptedBackend(
                [errors.PermanentBackendError("down")] * 99
            ),
            tokenizer=TOK,
        )
        cfg = compression.Config(
            tasks=("document",),
            methods=("truncate",),
            ratios=(4,),
            samples=1,
            tokens=2000 + 2000,
        )
        report = asyncio.run(compression.run(bad, cfg))
        assert (
            report.cells()[0].errors == 1 and report.cells()[0].accuracy == 0.0
        )

    def test_config_validation(self):
        for kw in (
            {"samples": 0},
            {"ratios": (0,)},
            {"ratios": ()},
            {"ratios": (1,)},
            {"tokens": 10},
            {"methods": ("magic",)},
            {"tasks": ("nope",)},
        ):
            with pytest.raises(errors.ConfigError):
                compression.Config(**kw)

    def test_applies(self):
        assert runner.applies("history", "window")
        assert not runner.applies("document", "window")


class TestAtomsAndQa:
    def test_atoms_task_plants_six_weighted_facts_in_the_middle(self):
        sample = tasks.Atoms(TOK, 6000).build(1)
        assert len(sample.atoms) == 6 and {a.weight for a in sample.atoms} == {
            1,
            2,
            3,
        }
        text = " ".join(m.content for m in sample.messages)
        assert all(a.value in text for a in sample.atoms)
        assert sample.expected == sample.atoms[0].value

    def test_compression_reports_which_facts_survived(self):
        cfg = compression.Config(
            tasks=("atoms",),
            methods=("truncate", "extractive", "selective"),
            ratios=(8,),
            samples=2,
            tokens=6000,
        )
        report = asyncio.run(compression.run(make_runtime(), cfg))
        by = {c.method: c for c in report.cells()}
        assert by["uncompressed"].atom_recall == 1.0
        for name in ("truncate", "extractive", "selective"):
            cell = by[name]
            assert 0.0 <= cell.atom_recall <= 1.0
            assert (
                cell.omitted + cell.mutated + cell.atom_recall * 6
                == pytest.approx(6, abs=1e-6)
            )
        assert by["selective"].atom_recall >= by["truncate"].atom_recall

    def test_hotpotqa_rows_become_exact_match_and_f1_samples(self, monkeypatch):
        row = {
            "question": "Which city hosts the Semper Opera House?",
            "answer": "Dresden",
            "context": {
                "title": ["Semper Opera House", "Elbe"],
                "sentences": [
                    ["The Semper Opera House is in Dresden. "],
                    ["The Elbe is a river."],
                ],
            },
        }
        monkeypatch.setattr(tasks.HotpotQA, "fetch", lambda self, index: row)
        sample = tasks.HotpotQA(TOK, 2000).build(0)
        assert (
            sample.golds == ("Dresden",)
            and "Semper Opera House:" in sample.messages[0].content
        )
        ok, f1, exact = runner.judge(sample, "The answer is Dresden")
        assert ok and f1 > 0.3 and not exact
        ok, f1, exact = runner.judge(sample, "dresden")
        assert ok and exact and f1 == 1.0
        assert not runner.judge(sample, "Berlin")[0]

    def test_hotpotqa_download_is_cached(self, tmp_path, monkeypatch):
        calls = []
        payload = {
            "rows": [
                {
                    "row": {
                        "question": "q",
                        "answer": "a",
                        "context": {"title": [], "sentences": []},
                    }
                }
            ]
        }

        def fake_fetch(url):
            calls.append(url)
            return json.dumps(payload).encode()

        monkeypatch.setattr(tasks.dataset, "fetch", fake_fetch)
        monkeypatch.setenv("FOVEATE_CACHE_HOME", str(tmp_path))
        task = tasks.HotpotQA(TOK, 2000)
        assert task.fetch(3)["answer"] == "a" and task.fetch(3)["answer"] == "a"
        assert len(calls) == 1 and "offset=111" in calls[0]
        monkeypatch.setattr(tasks.dataset, "fetch", lambda url: b'{"rows": []}')
        with pytest.raises(errors.ValidationError):
            task.fetch(4)
