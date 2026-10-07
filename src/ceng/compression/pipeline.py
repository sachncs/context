"""Composition of strategies: sequential pipelines and fallbacks."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ceng import errors
from ceng.compression import base, report

if TYPE_CHECKING:
    from ceng import context as context_lib


@dataclasses.dataclass(frozen=True)
class Pipeline(base.Compressor):
    """Runs stages in order, stopping as soon as the context fits.

    Intermediate stages are not budget-enforced; only the pipeline's final
    result is, so a cheap coarse stage can precede a precise one.

    Attributes:
        stages: Strategies to apply, in order (at least one).
    """

    stages: tuple[base.Compressor, ...] = ()
    name = "pipeline"

    def __post_init__(self) -> None:
        if not self.stages:
            raise errors.ConfigError("a pipeline needs at least one stage")

    @property
    def label(self) -> str:
        return "+".join(stage.label for stage in self.stages)

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        current = context
        for stage in self.stages:
            if current.token_count <= budget.tokens:
                break
            current = await stage.run(current, budget, trace)
        return current


@dataclasses.dataclass(frozen=True)
class Fallback(base.Compressor):
    """Uses `secondary` when `primary` fails with a backend error.

    Typical use: `Fallback(PartitionSummarizeCombine(), Extractive())` to
    degrade to offline compression when the LLM is unavailable.

    Attributes:
        primary: Preferred strategy.
        secondary: Strategy used if `primary` raises.
    """

    primary: base.Compressor
    secondary: base.Compressor
    name = "fallback"

    @property
    def label(self) -> str:
        return f"{self.primary.label}|{self.secondary.label}"

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        attempt = report.Trace()
        try:
            result = await self.primary.run(context, budget, attempt)
        except (errors.CompressionError, errors.BackendError) as exc:
            trace.add(
                report.StepRecord(
                    name=f"fallback after {type(exc).__name__}",
                    input_tokens=context.token_count,
                    output_tokens=context.token_count,
                )
            )
            return await self.secondary.run(context, budget, trace)
        for record in attempt.records:
            trace.add(record)
        return result
