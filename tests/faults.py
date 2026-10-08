"""Fault-injection backend for unit tests ONLY.

It replays scripted answers and exceptions so retry, timeout, circuit-breaker,
single-flight and error-isolation logic can be tested deterministically.
Everything that depends on real model behaviour is tested against a real
provider in `tests/integration/` instead.
"""

from __future__ import annotations

import asyncio
import collections
import threading
from collections.abc import Callable, Sequence

from foveate import usage as usage_lib
from foveate.backends import base

Step = str | BaseException | base.Completion | Callable[[base.Request], str]


class ScriptedBackend(base.Backend):
    """Replays scripted responses and records every request.

    Each call consumes the next step: a string is returned as the completion,
    an exception is raised (fault injection), and a callable is invoked with
    the request. When the script is exhausted, `default` is used.

    Attributes:
        requests: Every request received, in order.
    """

    def __init__(
        self,
        steps: Sequence[Step] = (),
        *,
        default: Callable[[base.Request], str] | None = None,
        delay: float = 0.0,
    ) -> None:
        """Creates the backend.

        Args:
            steps: Responses or faults to replay in order.
            default: Fallback once `steps` is exhausted. When omitted an
                exhausted script echoes a short deterministic digest of the
                last user message.
            delay: Simulated latency per call in seconds.
        """
        self.pending: collections.deque[Step] = collections.deque(steps)
        self.default = default or self.echo
        self.delay = delay
        self.requests: list[base.Request] = []
        self.lock = threading.Lock()

    @staticmethod
    def echo(request: base.Request) -> str:
        """Default responder: the first sentence-ish prefix of the input."""
        text = request.messages[-1].content.strip()
        return text[:60] or "ok"

    async def complete(self, request: base.Request) -> base.Completion:
        if self.delay:
            await asyncio.sleep(self.delay)
        with self.lock:
            self.requests.append(request)
            step = self.pending.popleft() if self.pending else self.default
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, base.Completion):
            return step
        text = step(request) if callable(step) else step
        prompt = sum(len(m.content) for m in request.messages) // 4
        return base.Completion(
            text=text,
            usage=usage_lib.Usage(prompt, max(1, len(text) // 4)),
            model=request.model,
        )
