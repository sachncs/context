"""Long-document benchmark: with and without foveate."""

from foveate.bench.longdoc.dataset import FinanceBench, Question
from foveate.bench.longdoc.gold import Example, GoldItem, starter_items
from foveate.bench.longdoc.metrics import Outcome
from foveate.bench.longdoc.pipelines import Pipeline, Result
from foveate.bench.longdoc.runner import (
    Config,
    Corpus,
    Report,
    StaticCorpus,
    run,
)

__all__ = [
    "Config",
    "Corpus",
    "Example",
    "FinanceBench",
    "GoldItem",
    "Outcome",
    "Pipeline",
    "Question",
    "Report",
    "Result",
    "StaticCorpus",
    "run",
    "starter_items",
]
