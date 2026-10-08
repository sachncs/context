"""Retry, timeout, circuit-breaker and concurrency wrapper for backends."""

from __future__ import annotations

import asyncio
import dataclasses
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from foveate import errors, observability
from foveate.backends import base
from foveate.internals import looplocal

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        attempts: Total tries including the first (>= 1).
        base_delay: Delay ceiling before the first retry, in seconds.
        max_delay: Cap on any single delay.
    """

    attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 20.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise errors.ConfigError("attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise errors.ConfigError("delays must be non-negative")

    def delay(
        self, attempt: int, rng: random.Random, hint: float | None
    ) -> float:
        """Returns the wait after failed `attempt` (1-based)."""
        ceiling = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        jittered = rng.uniform(0, ceiling)
        if hint is not None:
            return min(self.max_delay, max(jittered, hint))
        return jittered


async def retry_async(
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Runs `call`, retrying `TransientBackendError` with backoff.

    Args:
        call: Zero-argument coroutine factory.
        policy: Attempts and delays.
        sleep: Awaitable sleep (injectable for tests).
        rng: Jitter source.

    Returns:
        The first successful result.

    Raises:
        TransientBackendError: The last one, after `policy.attempts` tries.
    """
    jitter = rng or random.Random()
    last: errors.TransientBackendError | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await call()
        except errors.TransientBackendError as exc:
            last = exc
            if attempt < policy.attempts:
                await sleep(policy.delay(attempt, jitter, exc.retry_after))
    raise last if last is not None else errors.BackendError("no attempts")


class CircuitBreaker:
    """Opens after consecutive failures; half-opens after a cool-down."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Creates a closed breaker.

        Raises:
            ConfigError: If the threshold is below one.
        """
        if failure_threshold < 1:
            raise errors.ConfigError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        """Returns whether a call may proceed (closed or half-open probe)."""
        if self.opened_at is None:
            return True
        return self.clock() - self.opened_at >= self.reset_seconds

    def record_success(self) -> None:
        """Closes the breaker."""
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        """Counts a failure and opens the breaker at the threshold."""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()


class ResilientBackend(base.Backend):
    """Adds retries, per-call timeout, circuit breaking and a concurrency cap.

    Only `TransientBackendError` is retried. Provider exceptions are
    classified by `backends.classify` in the concrete backends before they
    reach this layer.
    """

    def __init__(
        self,
        inner: base.Backend,
        *,
        retry: RetryPolicy | None = None,
        timeout: float = 60.0,
        breaker: CircuitBreaker | None = None,
        max_concurrency: int = 8,
        rate_per_second: float | None = None,
        deadline: float | None = None,
        observers: tuple[observability.Observer, ...] = (),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wraps `inner`.

        Args:
            inner: Backend to protect.
            retry: Retry policy (default: 3 attempts).
            timeout: Per-call time limit in seconds.
            breaker: Circuit breaker (default: opens after 5 failures).
            max_concurrency: Maximum simultaneous calls.
            rate_per_second: Optional ceiling on call starts per second.
            deadline: Optional total seconds allowed across all retries of
                one request; a retry that would overrun it is not attempted.
            observers: Event sinks.
            sleep: Awaitable sleep (injectable for tests).
            rng: Jitter source.
            clock: Monotonic clock (injectable for tests).

        Raises:
            ConfigError: If a limit is not positive.
        """
        if timeout <= 0 or max_concurrency < 1:
            raise errors.ConfigError("timeout and max_concurrency must be > 0")
        if rate_per_second is not None and rate_per_second <= 0:
            raise errors.ConfigError("rate_per_second must be > 0")
        if deadline is not None and deadline <= 0:
            raise errors.ConfigError("deadline must be > 0")
        self.rate_per_second = rate_per_second
        self.deadline = deadline
        self.clock = clock
        self.next_start = 0.0
        self.pacing: looplocal.LoopLocal[asyncio.Lock] = looplocal.LoopLocal(
            asyncio.Lock
        )
        self.inner = inner
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self.breaker = breaker or CircuitBreaker()
        self.observers = observers
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.max_concurrency = max_concurrency
        self.semaphores: looplocal.LoopLocal[asyncio.Semaphore] = (
            looplocal.LoopLocal(lambda: asyncio.Semaphore(max_concurrency))
        )

    async def pace(self) -> None:
        """Spaces call starts to honour `rate_per_second`."""
        if self.rate_per_second is None:
            return
        async with self.pacing.get():
            now = self.clock()
            wait = self.next_start - now
            self.next_start = (
                max(now, self.next_start) + 1.0 / self.rate_per_second
            )
            if wait > 0:
                await self.sleep(wait)

    async def attempt(self, request: base.Request) -> base.Completion:
        """Performs one guarded call with timeout and breaker accounting."""
        if not self.breaker.allow():
            raise errors.CircuitOpenError("circuit breaker is open")
        await self.pace()
        try:
            async with self.semaphores.get():
                result = await asyncio.wait_for(
                    self.inner.complete(request), timeout=self.timeout
                )
        except asyncio.TimeoutError as exc:
            self.breaker.record_failure()
            raise errors.BackendTimeoutError(
                f"backend call exceeded {self.timeout}s"
            ) from exc
        except errors.TransientBackendError:
            self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return result

    async def complete(self, request: base.Request) -> base.Completion:
        last: errors.TransientBackendError | None = None
        started = self.clock()
        for attempt in range(1, self.retry.attempts + 1):
            try:
                return await self.attempt(request)
            except errors.CircuitOpenError:
                raise
            except errors.TransientBackendError as exc:
                last = exc
                if attempt == self.retry.attempts:
                    break
                delay = self.retry.delay(attempt, self.rng, exc.retry_after)
                if (
                    self.deadline is not None
                    and self.clock() - started + delay > self.deadline
                ):
                    break
                observability.emit(
                    self.observers,
                    observability.RetryScheduled(
                        source="backend",
                        attempt=attempt,
                        delay=delay,
                        reason=str(exc),
                    ),
                )
                await self.sleep(delay)
        raise last if last is not None else errors.BackendError("no attempts")

    async def aclose(self) -> None:
        await self.inner.aclose()
