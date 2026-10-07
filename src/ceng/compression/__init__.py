"""Compression strategies and the method-string resolver."""

from __future__ import annotations

from collections.abc import Mapping

from ceng import errors
from ceng.compression.base import Compressor
from ceng.compression.extractive import Extractive
from ceng.compression.hierarchical import Hierarchical
from ceng.compression.pipeline import Fallback, Pipeline
from ceng.compression.ppa import CombineMode, PartitionSummarizeCombine
from ceng.compression.report import (
    Budget,
    CompressionReport,
    Overflow,
    StepRecord,
    Trace,
)
from ceng.compression.truncate import Side, Truncate
from ceng.compression.ushape import MiddleMode, UShape
from ceng.compression.window import SlidingWindow


def resolve(method: str | Compressor, **options: object) -> Compressor:
    """Turns a method specification into a `Compressor`.

    Grammar: `a+b` builds a `Pipeline`, `a|b` a `Fallback` (lowest
    precedence, so `a+b|c` is `Fallback(Pipeline(a, b), c)`).

    Options apply to a single named method. For composite specs, pass each
    stage's options as a mapping keyed by stage name, e.g.
    `resolve("ushape+ppa", ppa={"leaf_tokens": 256})`.

    Args:
        method: A registered name, composite spec, or a ready instance.
        **options: Constructor options (see above).

    Returns:
        The configured strategy.

    Raises:
        ConfigError: For unknown methods or unknown/invalid options.
    """
    if isinstance(method, Compressor):
        if options:
            raise errors.ConfigError(
                "options cannot be combined with a Compressor instance"
            )
        return method
    if "|" in method or "+" in method:
        names = {p.strip() for p in method.replace("|", "+").split("+")}
        unknown = set(options) - names
        if unknown:
            raise errors.ConfigError(
                f"options for unknown stages: {sorted(unknown)}"
            )
    if "|" in method:
        parts = [part.strip() for part in method.split("|")]
        built = [resolve(part, **scoped(part, options)) for part in parts]
        result = built[0]
        for other in built[1:]:
            result = Fallback(result, other)
        return result
    if "+" in method:
        stages = [part.strip() for part in method.split("+")]
        return Pipeline(
            tuple(resolve(name, **scoped(name, options)) for name in stages)
        )
    cls = Compressor.registry.get(method.strip())
    try:
        return cls(**options)
    except TypeError as exc:
        raise errors.ConfigError(
            f"invalid options for {method!r}: {exc}"
        ) from exc


def scoped(name: str, options: Mapping[str, object]) -> dict[str, object]:
    """Returns the option mapping addressed to stage `name`."""
    value = options.get(name, {})
    if not isinstance(value, Mapping):
        raise errors.ConfigError(
            f"options for stage {name!r} must be a mapping, got {value!r}"
        )
    return dict(value)


__all__ = [
    "Budget",
    "CombineMode",
    "CompressionReport",
    "Compressor",
    "Extractive",
    "Fallback",
    "Hierarchical",
    "MiddleMode",
    "Overflow",
    "PartitionSummarizeCombine",
    "Pipeline",
    "Side",
    "SlidingWindow",
    "StepRecord",
    "Trace",
    "Truncate",
    "UShape",
    "resolve",
]
