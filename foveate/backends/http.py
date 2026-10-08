"""A small JSON-over-HTTP client built on the standard library.

Foveate talks to OpenAI-compatible servers (OpenAI, vLLM, Ollama, NVIDIA,
Together, Azure and many gateways) with `urllib`, so the core package needs no
SDK. Calls run in a worker thread so they never block the event loop, and
every failure is mapped onto Foveate's transient/permanent error taxonomy.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from foveate import errors

USER_AGENT = "foveate"
ERROR_BODY_CHARS = 300
TRANSIENT_STATUSES = (408, 409)


def retry_after(headers: Any) -> float | None:
    """Reads a `Retry-After` header given in seconds."""
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def from_status(
    status: int, message: str, delay: float | None
) -> errors.BackendError:
    """Maps an HTTP status onto the matching `BackendError`.

    429 is a rate limit; 408, 409 and 5xx are transient (worth retrying);
    every other status is permanent.
    """
    if status == 429:
        return errors.RateLimitError(message, retry_after=delay)
    if status in TRANSIENT_STATUSES or status >= 500:
        return errors.TransientBackendError(message, retry_after=delay)
    return errors.PermanentBackendError(message)


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POSTs `body` as JSON and returns the decoded JSON object (blocking).

    Raises:
        PermanentBackendError: For a non-http(s) URL or a 4xx reply.
        RateLimitError: For HTTP 429.
        TransientBackendError: For timeouts, connection problems and 5xx.
        ValidationError: If the reply is not a JSON object.
    """
    if not url.startswith(("https://", "http://")):
        raise errors.PermanentBackendError(f"refusing non-HTTP url: {url!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310
            raw = reply.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:ERROR_BODY_CHARS]
        raise from_status(
            exc.code,
            f"HTTP {exc.code} from {url}: {detail}",
            retry_after(exc.headers),
        ) from exc
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        OSError,
    ) as exc:
        raise errors.TransientBackendError(
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        decoded = json.loads(raw)
    except ValueError as exc:
        raise errors.ValidationError(
            f"reply from {url} is not JSON: {raw[:80]!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise errors.ValidationError(f"reply from {url} is not a JSON object")
    return decoded


async def apost_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Runs `post_json` in a worker thread."""
    return await asyncio.to_thread(post_json, url, body, headers, timeout)
