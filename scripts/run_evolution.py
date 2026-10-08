"""Evolve a playbook on part of a benchmark and test it on the rest.

    python scripts/run_evolution.py --task formula --train 10

Arms on the held-out samples: `baseline` (no playbook), `seed` (the curated
starting playbook) and `evolved` (what the evolution loop learned from the
training samples). The model comes from the FOVEATE_* environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

from foveate import Runtime
from foveate.bench import Arm, DDXPlus, Finer, Formula, Runner
from foveate.evolution import Evolver, EvolverConfig

TASKS = {"formula": Formula, "finer": Finer, "ddxplus": DDXPlus}


def main() -> int:
    """Runs the experiment and writes a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="formula")
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", default="results/evolution")
    args = parser.parse_args()
    benchmark = TASKS[args.task]()
    samples = benchmark.load_samples(args.train + args.test)
    train, test = samples[: args.train], samples[args.train :]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Runtime.from_env() as runtime:
        evolved = Evolver(
            runtime, EvolverConfig(max_reflector_rounds=2)
        ).evolve(
            benchmark.seed_playbook(),
            train,
            benchmark,
            epochs=args.epochs,
            checkpoint=out / "checkpoint.json",
        )
        print(
            f"evolved: {len(evolved.steps)} steps, train accuracy "
            f"{evolved.accuracy}, {len(evolved.playbook)} bullets"
        )
        arms = [
            Arm("baseline"),
            Arm("seed", benchmark.seed_playbook()),
            Arm("evolved", evolved.playbook),
        ]
        result = asyncio.run(Runner(runtime).arun(benchmark, test, arms))
    print(result.to_markdown())
    print("written:", result.write(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
