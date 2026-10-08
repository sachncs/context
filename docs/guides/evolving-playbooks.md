---
title: "Evolving a playbook"
description: "Let a model learn from its own mistakes: grow a written playbook of strategies, and test whether it helps on held-out questions."
---

A **playbook** is a short, structured list of strategies that you put in the prompt: how to approach this
kind of task, formulas to use, mistakes to avoid. Foveate can *evolve* one automatically, following the
Agentic Context Engineering method (Zhang et al., 2026, [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)).
The model's weights never change; only the text you give it does.

## The loop

For each training sample, three roles take turns:

1. **Generator** answers the question using the current playbook.
2. **Reflector** looks at a wrong answer and the ground truth, and explains what went wrong and which bullet
   would have helped or misled.
3. **Curator** edits the playbook: add a bullet, update one, merge duplicates, or delete a harmful one. Every
   bullet carries counters for how often it helped or hurt, and is trimmed when the playbook outgrows its
   token budget.

After the pass the playbook is plain text you can read, edit, save and reuse.

## Run it

```python
import asyncio

from foveate import Runtime
from foveate.bench import Arm, Formula, Runner
from foveate.evolution import Evolver, EvolverConfig

benchmark = Formula()                       # also: Finer, DDXPlus, or your own Benchmark
samples = benchmark.load_samples(20)
train, test = samples[:10], samples[10:]

with Runtime.from_env() as runtime:
    result = Evolver(runtime, EvolverConfig(max_reflector_rounds=2)).evolve(
        benchmark.seed_playbook(), train, benchmark, epochs=1,
        checkpoint="evolution.json",        # resume after a crash
    )
    arms = [
        Arm("baseline"),                                    # no playbook
        Arm("seed", benchmark.seed_playbook()),             # the curated start
        Arm("evolved", result.playbook),                    # what was learned
    ]
    report = asyncio.run(Runner(runtime).arun(benchmark, test, arms))
print(report.to_markdown())
print(result.playbook.render())             # the learned strategies, as text
```

`scripts/run_evolution.py` does exactly this and writes a report. The checkpoint is written after every step, so
a rate limit or a crash costs nothing already learned.

## Bring your own task

Subclass `foveate.bench.Benchmark` with `load_samples`, a seed playbook, and an `is_correct(predicted, target)` that
returns whether an answer matches. `Evolver.evolve` accepts any `Grader` and list of `Sample`s, so the benchmark class is
optional: only a way to tell right from wrong is required.

## What we measured

Three small tasks that ship with Foveate, answer model `gpt-oss-20b` (`reasoning_effort=low`), the playbook
evolved on 10 training samples and tested on 10 held-out ones:

| Task | No playbook | Starting playbook | Evolved playbook |
|---|---|---|---|
| Formula (financial formulas) | 70% | 80% | **90%** |
| FinER (financial entity tagging) | 80% | **100%** | **100%** |
| DDXPlus (diagnosis) | 100% | 100% | 100% |

Read this as a working demonstration, not a result. Ten held-out samples means one sample is ten points; the
gain on Formula is two questions. DDXPlus was already at the ceiling for this model. The evolved playbooks also
cost more input: the evolved arms sent roughly 7 to 20 times the baseline's prompt tokens (about 10,000 tokens
against 1,500 on Formula), which is the price of carrying the playbook. The loop is cheap to run and the
playbook is readable, so test it on your own task before relying on it.

## Cost and safety

* Each step makes up to `1 + max_reflector_rounds + 1` model calls, all cached, so a re-run is free.
* A step whose backend call fails is skipped and counted (`backend_errors`) unless you set `fail_fast=True`.
* Treat a learned playbook like code you did not write: read it before you ship it. It is text the model will
  obey.
