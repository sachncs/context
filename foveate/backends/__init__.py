"""LLM backends and the resilience layer around them."""

from foveate.backends.base import Backend, Completion, Request
from foveate.backends.none import NoBackend
from foveate.backends.providers import OpenAIBackend, VLLMBackend
from foveate.backends.resilient import (
    CircuitBreaker,
    ResilientBackend,
    RetryPolicy,
)

__all__ = [
    "Backend",
    "CircuitBreaker",
    "Completion",
    "NoBackend",
    "OpenAIBackend",
    "Request",
    "ResilientBackend",
    "RetryPolicy",
    "VLLMBackend",
]
