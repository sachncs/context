"""Writes summary.json next to a longdoc.json produced by an older run.

python scripts/summarize_results.py bench/results/run1
"""

from __future__ import annotations

import json
import pathlib
import sys

from foveate.bench.longdoc import gold, runner

GOLD_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "foveate/bench/longdoc/gold"
)


def main() -> int:
    """Summarises the run directory given on the command line."""
    directory = pathlib.Path(sys.argv[1])
    items = [
        item
        for name in ("answerable", "paraphrases", "unanswerable", "needles")
        for item in gold.read_items(GOLD_DIR / f"{name}.jsonl")
    ]
    report = runner.Report.load(directory / "longdoc.json", items)
    (directory / "summary.json").write_text(
        json.dumps(report.summary(), indent=1), encoding="utf-8"
    )
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
