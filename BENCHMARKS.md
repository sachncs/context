# Benchmarks

`ceng.bench` compares **arms** on a task: a *baseline* prompt, and a *ceng*
prompt that injects a playbook (the curated seed playbook, or one evolved by
`ceng.evolution`). Each sample is scored under every arm concurrently; backend
failures are counted separately and never scored as wrong answers.

```python
import asyncio
from ceng.bench import Arm, Finer, Formula, Runner
from ceng.evolution import Evolver

benchmark = Formula()
samples = benchmark.load_samples(limit=30)
seed = benchmark.seed_playbook()
evolved = Evolver(runtime).evolve(seed, samples, benchmark, epochs=2).playbook

result = asyncio.run(
    Runner(runtime).arun(
        benchmark, samples, [Arm("baseline"), Arm("seed", seed), Arm("evolved", evolved)]
    )
)
result.write(pathlib.Path("bench/results"))   # .json and .md
```

Reports are written to `bench/results/<name>-<timestamp>.{json,md}`. They keep
**Measured by ceng** and **Cited from the ACE paper** in separate sections.

## What the offline mode proves

`bench.offline.wiring_runtime(samples)` builds a scripted model that answers correctly only when a playbook
is present in the prompt. A non-zero delta therefore proves the playbook
reaches the model; a zero delta (for an empty playbook arm) proves it does
not leak in otherwise. It says nothing about real model quality, and its model
is named `offline-wiring-check` in the report.

## Tasks

| Name | Shape | Grading |
|---|---|---|
| `finer` | Token tagging with BIO labels | Exact label sequence (token prefix tolerated) |
| `formula` | Numeric financial questions | Relative tolerance; `$`, thousands separators and `%` handled |
| `ddxplus` | 4-way diagnosis, answer is an option index | First integer in the reply |

The packaged fixtures (`src/ceng/bench/fixtures/`) hold 20 rows each. They are
smoke data, and the `finer` fixture uses 9 CoNLL-style tags, not the 139 XBRL
types of FiNER-139. For serious evaluation load real data with
`Benchmark.load_samples(path=...)`.

## Published numbers

This release publishes **no measured numbers**. The v1 reports in earlier
versions compared the model with itself (the "ceng" arm never contained a
playbook) and are withdrawn. The ACE paper's figures (FiNER 70.7 to 78.3,
Formula 67.5 to 85.5, DDXPlus 75.2 to 90.2, DeepSeek-V3.1) are cited in reports
as context only: different model, full datasets.
