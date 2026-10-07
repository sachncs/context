"""Structured events and observers.

Library code emits typed `Event` objects to the observers configured on the
`Runtime`. Observers decide what to do with them (log, aggregate metrics).
The library itself only ever installs a `logging.NullHandler`.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import threading
from collections.abc import Sequence

from ceng import usage as usage_lib

logger = logging.getLogger("ceng")
logger.addHandler(logging.NullHandler())


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """Base class of all events.

    Attributes:
        source: Component that emitted the event (e.g. "ppa").
    """

    source: str


@dataclasses.dataclass(frozen=True, slots=True)
class StepFinished(Event):
    """A named processing step completed.

    Attributes:
        step: Step label, e.g. "leaf 3".
        seconds: Wall-clock duration.
    """

    step: str = ""
    seconds: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class CacheLookup(Event):
    """A cache lookup finished.

    Attributes:
        hit: Whether a value was found.
    """

    hit: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class BackendCall(Event):
    """A backend call completed.

    Attributes:
        model: Model name.
        usage: Token usage reported or estimated.
        seconds: Wall-clock latency.
    """

    model: str = ""
    usage: usage_lib.Usage = dataclasses.field(default_factory=usage_lib.Usage)
    seconds: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class RetryScheduled(Event):
    """A failed call will be retried.

    Attributes:
        attempt: 1-based number of the attempt that just failed.
        delay: Seconds until the next attempt.
        reason: Short description of the failure.
    """

    attempt: int = 0
    delay: float = 0.0
    reason: str = ""


class Observer(abc.ABC):
    """Receives events. Implementations must be thread-safe and not raise."""

    @abc.abstractmethod
    def handle(self, event: Event) -> None:
        """Processes one event."""


class LoggingObserver(Observer):
    """Writes events to the standard `logging` hierarchy."""

    def __init__(self, level: int = logging.INFO) -> None:
        self.level = level

    def handle(self, event: Event) -> None:
        logger.log(self.level, "%s", event)


class MetricsObserver(Observer):
    """Aggregates cache, usage, retry and latency counters."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.backend_calls = 0
        self.retries = 0
        self.usage = usage_lib.Usage()
        self.backend_seconds = 0.0

    def handle(self, event: Event) -> None:
        with self.lock:
            if isinstance(event, CacheLookup):
                if event.hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
            elif isinstance(event, BackendCall):
                self.backend_calls += 1
                self.usage = self.usage + event.usage
                self.backend_seconds += event.seconds
            elif isinstance(event, RetryScheduled):
                self.retries += 1


def emit(observers: Sequence[Observer], event: Event) -> None:
    """Delivers `event` to every observer, isolating observer failures."""
    for observer in observers:
        try:
            observer.handle(event)
        except Exception:
            logger.exception("observer %r failed", observer)
