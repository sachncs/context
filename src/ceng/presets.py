"""Named presets mirroring the ACE paper's recommended hyperparameter sweeps.

Use these instead of hand-rolling ``EvolverConfig`` instances:

    from ceng.presets import FinerOffline5Rounds
    from ceng.playbook.evolver import Evolver

    ev = Evolver(llm="gpt-4o-mini", config=FinerOffline5Rounds())
    pb, stats = ev.run(...)

All presets use:
- ``max_reflector_rounds = 5``           (paper §A.6 best)
- ``dedup_threshold = 0.90``              (paper §A.6 best)
- ``playbook_token_budget = 80_000``      (upstream ACE default)
- ``curator_frequency = 1``              (run curator every step)

They differ in ``use_ground_truth`` (offline = True, online = False)
to match the paper's two experimental modes.
"""

from __future__ import annotations

from ceng.playbook.evolver import EvolverConfig


FinerOffline5Rounds = EvolverConfig(
    max_reflector_rounds=5,
    dedup_threshold=0.90,
    playbook_token_budget=80_000,
    curator_frequency=1,
    use_ground_truth=True,
)

FinerOnline5Rounds = EvolverConfig(
    max_reflector_rounds=5,
    dedup_threshold=0.90,
    playbook_token_budget=80_000,
    curator_frequency=1,
    use_ground_truth=False,
)

FormulaOffline5Rounds = FinerOffline5Rounds  # same config, different seed
FormulaOnline5Rounds = FinerOnline5Rounds

DDXPlusOffline5Rounds = FinerOffline5Rounds
DDXPlusOnline5Rounds = FinerOnline5Rounds

AppWorldOnlineFineTune = EvolverConfig(
    max_reflector_rounds=5,
    dedup_threshold=0.90,
    playbook_token_budget=80_000,
    curator_frequency=1,
    use_ground_truth=False,
)

# A barebones config for tiny smoke runs.
Smoke3Rounds = EvolverConfig(
    max_reflector_rounds=3,
    dedup_threshold=0.90,
    playbook_token_budget=20_000,
    curator_frequency=1,
    use_ground_truth=True,
)


__all__ = [
    "FinerOffline5Rounds",
    "FinerOnline5Rounds",
    "FormulaOffline5Rounds",
    "FormulaOnline5Rounds",
    "DDXPlusOffline5Rounds",
    "DDXPlusOnline5Rounds",
    "AppWorldOnlineFineTune",
    "Smoke3Rounds",
]
