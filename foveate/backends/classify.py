"""Maps provider exceptions onto foveate's transient/permanent taxonomy."""

from __future__ import annotations

from foveate import errors

TRANSIENT_NAME_HINTS = (
    "timeout",
    "connection",
    "unavailable",
    "overloaded",
    "internalserver",
)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extracts a Retry-After header value from a provider exception."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def classify(exc: BaseException) -> errors.BackendError:
    """Wraps a provider exception in the matching `BackendError`.

    HTTP status wins when present (429 rate limit; 408/409/5xx transient;
    other 4xx permanent). Otherwise class names are inspected for timeout and
    connection hints. Anything unrecognised is permanent so that programming
    errors are never retried.

    Args:
        exc: The exception raised by a provider SDK.

    Returns:
        A `BackendError` subclass with `exc` available as `__cause__`
        once raised via `raise classify(exc) from exc`.
    """
    if isinstance(exc, errors.BackendError):
        return exc
    message = f"{type(exc).__name__}: {exc}"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return errors.RateLimitError(
                message, retry_after=retry_after_seconds(exc)
            )
        if status in (408, 409) or status >= 500:
            return errors.TransientBackendError(
                message, retry_after=retry_after_seconds(exc)
            )
        return errors.PermanentBackendError(message)
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return errors.RateLimitError(
            message, retry_after=retry_after_seconds(exc)
        )
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        hint in name for hint in TRANSIENT_NAME_HINTS
    ):
        return errors.TransientBackendError(message)
    return errors.PermanentBackendError(message)
