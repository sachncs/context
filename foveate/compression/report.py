"""Budget, step records and the compression report."""

from __future__ import annotations

import dataclasses
import enum
import threading
from collections.abc import Mapping
from typing import Any

from foveate import errors
from foveate import usage as usage_lib


class Overflow(str, enum.Enum):
    """What to do when a strategy cannot reach the budget."""

    RAISE = "raise"
    TRUNCATE = "truncate"


@dataclasses.dataclass(frozen=True, slots=True)
class Budget:
    """A token ceiling for a whole context.

    Attributes:
        tokens: Maximum total content tokens (> 0).
        overflow: Policy when the strategy's output still exceeds `tokens`.
    """

    tokens: int
    overflow: Overflow = Overflow.RAISE

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise errors.ConfigError("budget must be >= 1 token")

    @classmethod
    def of(cls, value: int | Budget) -> Budget:
        """Coerces an int or a `Budget` into a `Budget`."""
        return value if isinstance(value, Budget) else cls(value)


@dataclasses.dataclass(frozen=True, slots=True)
class StepRecord:
    """One unit of work performed by a strategy.

    Attributes:
        name: Step label, e.g. "leaf 3" or "combine".
        input_tokens: Tokens consumed by the step.
        output_tokens: Tokens produced by the step.
        cached: Whether an LLM answer was served from the cache.
        usage: Provider-reported token usage (zero when cached or LLM-free).
        seconds: Wall-clock duration.
    """

    name: str
    input_tokens: int
    output_tokens: int
    cached: bool = False
    usage: usage_lib.Usage = dataclasses.field(default_factory=usage_lib.Usage)
    seconds: float = 0.0

    def to_mapping(self) -> dict[str, object]:
        """Returns a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached": self.cached,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "seconds": self.seconds,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StepRecord:
        """Rebuilds a record from `to_mapping` output.

        Raises:
            ValidationError: If a field is missing or mistyped.
        """
        try:
            return cls(
                name=str(data["name"]),
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
                cached=bool(data["cached"]),
                usage=usage_lib.Usage(
                    int(data["prompt_tokens"]), int(data["completion_tokens"])
                ),
                seconds=float(data["seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise errors.ValidationError(f"invalid step record: {exc}") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class CompressionReport:
    """Provenance of a compression.

    Attributes:
        method: Strategy name (pipelines are joined with "+").
        version: Strategy version string.
        budget: Requested token ceiling.
        original_tokens: Tokens before compression.
        final_tokens: Tokens after compression.
        steps: Ordered step records.
        cost_usd: Estimated spend from the runtime's price table.
        truncated: Whether hard truncation was needed to meet the budget.
        seconds: Total wall-clock duration.
    """

    method: str
    version: str
    budget: int
    original_tokens: int
    final_tokens: int
    steps: tuple[StepRecord, ...] = ()
    cost_usd: float = 0.0
    truncated: bool = False
    seconds: float = 0.0

    def to_mapping(self) -> dict[str, object]:
        """Returns a JSON-serialisable mapping."""
        return {
            "method": self.method,
            "version": self.version,
            "budget": self.budget,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "steps": [step.to_mapping() for step in self.steps],
            "cost_usd": self.cost_usd,
            "truncated": self.truncated,
            "seconds": self.seconds,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CompressionReport:
        """Rebuilds a report from `to_mapping` output.

        Raises:
            ValidationError: If a field is missing or mistyped.
        """
        try:
            return cls(
                method=str(data["method"]),
                version=str(data["version"]),
                budget=int(data["budget"]),
                original_tokens=int(data["original_tokens"]),
                final_tokens=int(data["final_tokens"]),
                steps=tuple(
                    StepRecord.from_mapping(step) for step in data["steps"]
                ),
                cost_usd=float(data["cost_usd"]),
                truncated=bool(data["truncated"]),
                seconds=float(data["seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise errors.ValidationError(f"invalid report: {exc}") from exc

    @property
    def usage(self) -> usage_lib.Usage:
        """Returns summed provider usage across steps."""
        total = usage_lib.Usage()
        for step in self.steps:
            total = total + step.usage
        return total

    @property
    def cache_hits(self) -> int:
        """Returns the number of cached LLM steps."""
        return sum(1 for step in self.steps if step.cached)

    @property
    def ratio(self) -> float:
        """Returns final / original tokens (1.0 for empty input)."""
        if self.original_tokens == 0:
            return 1.0
        return self.final_tokens / self.original_tokens


class Trace:
    """Thread-safe collector of step records for one compression run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.records: list[StepRecord] = []
        self.truncated = False

    def add(self, record: StepRecord) -> None:
        """Appends a step record."""
        with self.lock:
            self.records.append(record)
