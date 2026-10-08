"""Long-document benchmark: with and without foveate."""

from foveate.bench.longdoc.dataset import FinanceBench, Question
from foveate.bench.longdoc.gold import GoldItem
from foveate.bench.longdoc.metrics import Outcome
from foveate.bench.longdoc.pipelines import Pipeline, Result
from foveate.bench.longdoc.runner import Config, Corpus, Report, run

__all__ = [
    "Config",
    "Corpus",
    "FinanceBench",
    "GoldItem",
    "Outcome",
    "Pipeline",
    "Question",
    "Report",
    "Result",
    "run",
]
