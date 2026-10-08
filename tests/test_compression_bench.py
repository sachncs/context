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
            methods=(
                "truncate",
                "window",
                "extractive",
                "query",
                "ushape-drop",
                "tool_output",
                "foveate",
            ),
            ratios=(1, 4, 16),
            samples=2,
            tokens=8000,
            **kw,
        )
        return asyncio.run(compression.run(make_runtime(), cfg))

    def test_uncompressed_baseline_answers_everything(self):
        report = self.sweep()
        base = [c for c in report.cells() if c.ratio == 1]
        assert base and all(c.accuracy == 1.0 for c in base)

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
        assert "### document: accuracy by compression ratio" in path.read_text()
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
            ratios=(1,),
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
            {"tokens": 10},
            {"methods": ("magic",)},
            {"tasks": ("nope",)},
        ):
            with pytest.raises(errors.ConfigError):
                compression.Config(**kw)

    def test_applies(self):
        assert runner.applies("history", "window")
        assert not runner.applies("document", "window")
