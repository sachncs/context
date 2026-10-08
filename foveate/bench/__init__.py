"""Benchmarks.

* Playbook benchmark (this package): `Benchmark`, `Runner` and the `Arm`s you
  compare, with tasks `Finer`, `Formula` and `DDXPlus`; used to test playbooks
  from `foveate.evolution`.
* `foveate.bench.longdoc`: long documents, gold sets and pipelines.
* `foveate.bench.needle`: needle-in-a-haystack grids.
* `foveate.bench.compression`: accuracy against compression ratio.
* `foveate.bench.scoring`: exact match, F1 and atom recall.
"""

from foveate.bench.base import Benchmark
from foveate.bench.runner import Arm, ArmResult, BenchResult, Runner
from foveate.bench.tasks import DDXPlus, Finer, Formula

__all__ = [
    "Arm",
    "ArmResult",
    "BenchResult",
    "Benchmark",
    "DDXPlus",
    "Finer",
    "Formula",
    "Runner",
]
