"""Runs the compression sweep (a dev tool).

    python scripts/run_compression.py --samples 6
    python scripts/run_compression.py --tasks document --methods truncate,query

The answer model comes from the FOVEATE_* environment variables. Results go
to results/compression/.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

from foveate import Runtime
from foveate.bench import compression

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    """Parses arguments and runs the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="history,atoms,tool,document,qa")
    parser.add_argument("--methods", default=",".join(compression.METHODS))
    parser.add_argument("--ratios", default="1,2,4,8,16")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "results/compression"))
    args = parser.parse_args()
    config = compression.Config(
        tasks=tuple(args.tasks.split(",")),
        methods=tuple(args.methods.split(",")),
        ratios=tuple(int(r) for r in args.ratios.split(",")),
        samples=args.samples,
        tokens=args.tokens,
        concurrency=args.concurrency,
    )
    with Runtime.from_env() as runtime:
        report = asyncio.run(compression.run(runtime, config, print))
    print(report.to_markdown())
    print("written:", report.write(pathlib.Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
