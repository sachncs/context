"""Learn a playbook from a handful of graded samples, then read it.

Needs a model (see common.py). Uses the small built-in Formula benchmark.
"""

import common

from foveate.bench import Formula
from foveate.evolution import Evolver, EvolverConfig


def main() -> None:
    if not common.has_model():
        print("Set FOVEATE_MODEL (see common.py) to run the evolution loop.")
        return
    benchmark = Formula()
    samples = benchmark.load_samples(4)
    with common.model_runtime() as runtime:
        result = Evolver(runtime, EvolverConfig(max_reflector_rounds=1)).evolve(
            benchmark.seed_playbook(), samples, benchmark
        )
    print(
        f"steps: {len(result.steps)}, accuracy while learning: {result.accuracy}"
    )
    print(result.playbook.render()[:600])


if __name__ == "__main__":
    main()
