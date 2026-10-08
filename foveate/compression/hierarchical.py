"""Repeated PPA rounds until the budget is met."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from foveate import errors
from foveate.compression import base, ppa, report

if TYPE_CHECKING:
    from foveate import context as context_lib


@base.Compressor.register("hierarchical")
@dataclasses.dataclass(frozen=True)
class Hierarchical(ppa.PartitionSummarizeCombine):
    """Runs PPA repeatedly, feeding each result into the next round.

    Stops when the budget is met, when a round makes no progress, or after
    `max_rounds`. This is what closes the gap when one pass of summaries is
    still larger than the budget.

    Attributes:
        max_rounds: Upper bound on PPA rounds (>= 1).
    """

    max_rounds: int = 4

    def __post_init__(self) -> None:
        ppa.PartitionSummarizeCombine.__post_init__(self)
        if self.max_rounds < 1:
            raise errors.ConfigError("max_rounds must be >= 1")

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        current = context
        for round_number in range(self.max_rounds):
            before = current.token_count
            current = await ppa.PartitionSummarizeCombine.run(
                self, current, budget, trace
            )
            after = current.token_count
            trace.add(
                report.StepRecord(
                    name=f"round {round_number + 1}",
                    input_tokens=before,
                    output_tokens=after,
                )
            )
            if after <= budget.tokens or after >= before:
                break
        return current
