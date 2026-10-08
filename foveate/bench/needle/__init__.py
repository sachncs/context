"""Needle-in-a-haystack benchmark: length x depth grids for every pipeline."""

from foveate.bench.needle.cases import Case, grid, literal, multi, nonliteral
from foveate.bench.needle.runner import Cell, Config, Outcome, Report, run
from foveate.bench.needle.sources import (
    BookSource,
    ProseSource,
    Source,
    load_nolima,
)

__all__ = [
    "BookSource",
    "Case",
    "Cell",
    "Config",
    "Outcome",
    "ProseSource",
    "Report",
    "Source",
    "grid",
    "literal",
    "load_nolima",
    "multi",
    "nonliteral",
    "run",
]
