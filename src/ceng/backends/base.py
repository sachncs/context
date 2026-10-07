"""Backend interface and request/response types."""

from __future__ import annotations

import abc
import dataclasses

from ceng import messages as messages_lib
from ceng import usage as usage_lib
from ceng.internals import hashing, registry


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    """A chat-completion request.

    Attributes:
        model: Model identifier understood by the backend.
        messages: Conversation to complete.
        temperature: Sampling temperature.
        max_tokens: Completion cap, or None for the backend default.
    """

    model: str
    messages: tuple[messages_lib.Message, ...]
    temperature: float = 0.0
    max_tokens: int | None = None

    @property
    def fingerprint(self) -> str:
        """Returns a digest covering every field that affects the output."""
        return hashing.fingerprint(
            self.model,
            [m.to_mapping() for m in self.messages],
            self.temperature,
            self.max_tokens,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Completion:
    """A model response.

    Attributes:
        text: Generated text.
        usage: Token usage, as reported or estimated.
        model: Model that served the request.
        seconds: Wall-clock latency (0.0 for cache hits).
        cached: Whether the response came from the cache.
    """

    text: str
    usage: usage_lib.Usage = dataclasses.field(default_factory=usage_lib.Usage)
    model: str = ""
    seconds: float = 0.0
    cached: bool = False


class Backend(abc.ABC):
    """An asynchronous LLM provider."""

    registry: registry.Registry[type[Backend]] = registry.Registry("backend")

    @abc.abstractmethod
    async def complete(self, request: Request) -> Completion:
        """Sends `request` and returns the completion.

        Raises:
            TransientBackendError: For failures worth retrying.
            PermanentBackendError: For failures that retrying cannot fix.
        """

    async def aclose(self) -> None:
        """Releases any held resources. Default: nothing to release."""
