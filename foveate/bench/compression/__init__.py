"""Compression benchmark: accuracy as the context shrinks."""

from foveate.bench.compression.runner import (
    METHODS,
    RATIOS,
    Cell,
    Config,
    Report,
    Row,
    run,
)
from foveate.bench.compression.tasks import Sample, Task

__all__ = [
    "METHODS",
    "RATIOS",
    "Cell",
    "Config",
    "Report",
    "Row",
    "Sample",
    "Task",
    "run",
]
