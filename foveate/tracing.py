"""OpenTelemetry tracing for foveate events.

`TracingObserver` turns runtime events into spans on any tracer that follows
the OpenTelemetry API, so foveate calls show up next to the rest of an
application's traces. The `opentelemetry-api` package is not imported here:
pass `opentelemetry.trace.get_tracer("foveate")` (or any compatible tracer).

    from opentelemetry import trace
    runtime = Runtime(observers=[TracingObserver(trace.get_tracer("foveate"))])
"""

from __future__ import annotations

import time
from typing import Protocol

from foveate import observability

NANOSECONDS = 1_000_000_000


class Span(Protocol):
    """The slice of the OpenTelemetry span API that is used."""

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        """Sets one attribute."""

    def end(self, end_time: int | None = None) -> None:
        """Ends the span at `end_time` (epoch nanoseconds)."""


class Tracer(Protocol):
    """The slice of the OpenTelemetry tracer API that is used."""

    def start_span(self, name: str, *, start_time: int | None = None) -> Span:
        """Starts a span at `start_time` (epoch nanoseconds)."""


class TracingObserver(observability.Observer):
    """Records backend calls, retries, steps and cache lookups as spans.

    Events are reported after the fact, so each span is back-dated by the
    event's own duration. Attribute names follow the GenAI conventions
    (`gen_ai.request.model`, `gen_ai.usage.*`) where one exists.
    """

    def __init__(self, tracer: Tracer) -> None:
        """Creates the observer.

        Args:
            tracer: An OpenTelemetry-compatible tracer.
        """
        self.tracer = tracer

    def handle(self, event: observability.Event) -> None:
        now = time.time_ns()
        if isinstance(event, observability.BackendCall):
            self.record(
                f"foveate.backend {event.model}",
                now,
                event.seconds,
                {
                    "foveate.source": event.source,
                    "gen_ai.request.model": event.model,
                    "gen_ai.usage.input_tokens": event.usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": event.usage.completion_tokens,
                },
            )
        elif isinstance(event, observability.StepFinished):
            self.record(
                f"foveate.{event.source}.{event.step}",
                now,
                event.seconds,
                {"foveate.source": event.source, "foveate.step": event.step},
            )
        elif isinstance(event, observability.RetryScheduled):
            self.record(
                "foveate.retry",
                now,
                0.0,
                {
                    "foveate.source": event.source,
                    "foveate.attempt": event.attempt,
                    "foveate.delay_seconds": event.delay,
                    "foveate.reason": event.reason,
                },
            )
        elif isinstance(event, observability.CacheLookup):
            self.record(
                "foveate.cache",
                now,
                0.0,
                {
                    "foveate.source": event.source,
                    "foveate.cache_hit": event.hit,
                },
            )

    def record(
        self,
        name: str,
        end_ns: int,
        seconds: float,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        """Emits one finished span of the given duration."""
        start_ns = end_ns - int(seconds * NANOSECONDS)
        span = self.tracer.start_span(name, start_time=start_ns)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.end(end_time=end_ns)
