"""Agentic Context Engineering: the evolving playbook (arXiv:2510.04618)."""

from foveate.evolution.evolver import (
    Checkpoint,
    EvolutionResult,
    Evolver,
    EvolverConfig,
    StepStats,
)
from foveate.evolution.grading import Grader, Sample
from foveate.evolution.operations import (
    AddOp,
    CuratorOp,
    DeleteOp,
    MergeOp,
    UpdateOp,
)
from foveate.evolution.playbook import Bullet, Playbook
from foveate.evolution.roles import Curator, Generator, Reflector

__all__ = [
    "AddOp",
    "Bullet",
    "Checkpoint",
    "Curator",
    "CuratorOp",
    "DeleteOp",
    "EvolutionResult",
    "Evolver",
    "EvolverConfig",
    "Generator",
    "Grader",
    "MergeOp",
    "Playbook",
    "Reflector",
    "Sample",
    "StepStats",
    "UpdateOp",
]
