"""Macro-fallacy check and ACE playbook evolution with a real model."""

import common

from ceng import Context, Message, Role
from ceng.bench import Formula
from ceng.evolution import Evolver, EvolverConfig


def main() -> None:
    with common.model_runtime() as runtime:
        context = Context((Message(Role.USER, "probe"),), runtime)
        verdict = context.verify(
            "macro_fallacy",
            question="What fraction of adults own a bicycle?",
            population="adults in the Netherlands",
            tree=[
                {"description": "city dwellers", "prior": 0.4},
                {"description": "town and village dwellers", "prior": 0.6},
            ],
            tolerance=0.5,
        )
        print("consistent:", verdict.passed, verdict.detail)

        benchmark = Formula()
        result = Evolver(runtime, EvolverConfig(max_reflector_rounds=1)).evolve(
            benchmark.seed_playbook(), benchmark.load_samples(3), benchmark
        )
        print("accuracy:", result.accuracy, "bullets:", len(result.playbook))


if __name__ == "__main__":
    main()
