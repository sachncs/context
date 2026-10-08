"""Hard truncation: the cheapest, LLM-free strategy."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

from foveate.compression import base, fit, report

if TYPE_CHECKING:
    from foveate import context as context_lib


class Side(str, enum.Enum):
    """Which part of an oversize message survives truncation."""

    HEAD = "head"
    TAIL = "tail"
    MIDDLE = "middle"


@base.Compressor.register("truncate")
@dataclasses.dataclass(frozen=True)
class Truncate(base.Compressor):
    """Cuts the largest messages until the context fits.

    Attributes:
        keep: Which part of an oversize message to keep.
    """

    keep: Side = Side.TAIL

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        fitted = fit.fit(
            context.messages,
            budget.tokens,
            context.runtime.tokenizer,
            side=Side(self.keep).value,
        )
        return dataclasses.replace(context, messages=fitted)
