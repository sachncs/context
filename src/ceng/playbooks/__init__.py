"""Curated seed playbooks for the public benchmarks ceng ships with.

The seed is the most important factor in ceng's measured accuracy:
the ACE paper showed that a non-empty starting playbook is the
difference between the +5 baseline and the +15 published deltas.

Each ``seed_<benchmark>`` function returns a Playbook-ready text.
Use them directly::

    from ceng.playbook import parse_playbook
    from ceng.eval.finer import seed_playbook
    pb = parse_playbook(seed_playbook())
    ev.run(playbook=pb, ...)

Or via the ``ceng.playbooks`` package:

    from ceng.playbooks import finer_seed
    pb = parse_playbook(finer_seed())

These seeds are intentionally conservative (no API keys, no model
references); they capture the *task shape* and a few domain
heuristics. The Evolver grows them from there.
"""

from __future__ import annotations


def finer_seed() -> str:
    """Curated XBRL-aware seed playbook for FiNER (token tagging)."""
    from ceng.eval.finer import seed_playbook as _seed

    return _seed()


def formula_seed() -> str:
    """Curated financial-formula seed for Formula (numeric computation)."""
    from ceng.eval.formula import seed_playbook as _seed

    return _seed()


def ddxplus_seed() -> str:
    """Curated medical-reasoning seed for DDXPlus (4-way diagnosis)."""
    from ceng.eval.ddxplus import seed_playbook as _seed

    return _seed()


__all__ = ["finer_seed", "formula_seed", "ddxplus_seed"]
