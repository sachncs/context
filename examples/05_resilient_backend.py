"""Wrap any backend in retries, a timeout, a circuit breaker and a deadline.

The endpoint here is unreachable on purpose: the error is classified as
transient, retried with backoff, and then raised as one clear exception.
"""

import asyncio

from foveate import Message, Role, errors
from foveate.backends import OpenAIBackend, ResilientBackend, RetryPolicy
from foveate.backends.base import Request


async def main() -> None:
    backend = ResilientBackend(
        OpenAIBackend(base_url="http://127.0.0.1:9/v1", timeout=1),
        retry=RetryPolicy(attempts=2, base_delay=0.1, max_delay=0.2),
        timeout=5,
        max_concurrency=4,
    )
    request = Request(messages=(Message(Role.USER, "hi"),), model="any")
    try:
        await backend.complete(request)
    except errors.BackendError as exc:
        print(type(exc).__name__, "-", str(exc)[:80])


if __name__ == "__main__":
    asyncio.run(main())
