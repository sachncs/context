"""Exception hierarchy for ceng.

Every error raised deliberately by this library derives from `CengError`, so
callers can catch the whole family with a single clause. Backend errors are
split into transient (safe to retry) and permanent (never retried).
"""

from __future__ import annotations


class CengError(Exception):
    """Base class for all errors raised by ceng."""


class ConfigError(CengError):
    """Raised when configuration or options are invalid."""


class ValidationError(CengError):
    """Raised when data (input, model output, files) fails validation."""


class BudgetExceededError(CengError):
    """Raised when a context cannot be brought within its token budget.

    Attributes:
        budget: The token budget that was requested.
        actual: The token count that was achieved.
    """

    def __init__(self, budget: int, actual: int) -> None:
        super().__init__(
            f"could not fit context in budget: {actual} tokens > {budget}"
        )
        self.budget = budget
        self.actual = actual


class CompressionError(CengError):
    """Raised when a compression step fails.

    Attributes:
        step: Human-readable name of the failing step (e.g. "leaf 4").
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        step: str = "",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.cause = cause


class BackendError(CengError):
    """Base class for failures talking to an LLM backend."""


class TransientBackendError(BackendError):
    """A failure that may succeed on retry (network, 5xx, overload).

    Attributes:
        retry_after: Server-suggested delay in seconds, if provided.
    """

    def __init__(
        self, message: str, *, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(TransientBackendError):
    """The backend rejected the request due to rate limiting."""


class BackendTimeoutError(TransientBackendError):
    """A backend call exceeded its time limit."""


class PermanentBackendError(BackendError):
    """A failure that retrying cannot fix (auth, bad request, not found)."""


class CircuitOpenError(TransientBackendError):
    """The circuit breaker is open; the call was rejected without being sent."""
