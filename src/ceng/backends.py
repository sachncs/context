"""LLM backend adapters behind a single callable interface.

Every backend implements :class:`Backend` with one method,
:meth:`Backend.complete`, that takes an OpenAI-style ``messages`` list,
a ``model`` identifier, and arbitrary backend-specific kwargs (typically
``temperature``, ``max_tokens``, ``stop``), and returns the assistant's
text response.

Three adapters ship in this module:

* :class:`LiteLLMBackend` — default; routes through ``litellm`` so a
  single code path works for hosted vLLM (OpenAI-compatible HTTP),
  OpenAI, Anthropic, Bedrock, etc.
* :class:`VLLMBackend` — in-process ``vllm.LLM`` for users running
  vLLM directly on a GPU machine without an HTTP server.
* :class:`OpenAIBackend` — raw ``openai.OpenAI`` client; for callers
  who don't want ``litellm`` as a runtime dependency.

Heavy third-party imports (``litellm``, ``vllm``, ``openai``) are
performed lazily inside :meth:`complete` so the package stays importable
even when an extra isn't installed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


ENV_BACKEND = "CENG_BACKEND"
DEFAULT_BACKEND = "litellm"


@runtime_checkable
class Backend(Protocol):
    """Callable interface every adapter implements."""

    name: str

    def complete(self, messages: list[dict], model: str, **kw: Any) -> str:
        """Return the assistant's text for ``messages`` using ``model``."""
        ...


@dataclass
class LiteLLMBackend:
    """Routes completions through ``litellm.completion``."""

    name: str = "litellm"

    def complete(self, messages: list[dict], model: str, **kw: Any) -> str:
        """Send ``messages`` to ``model`` via ``litellm`` and return text.

        Args:
            messages: OpenAI-style chat messages list.
            model: Any model id ``litellm`` understands (e.g.
                ``"gpt-4o-mini"`` for OpenAI,
                ``"hosted_vllm/llama-3-8b"`` for a vLLM HTTP server
                with ``OPENAI_API_BASE`` set, etc.).
            **kw: Forwarded to ``litellm.completion``. The common keys
                are ``temperature``, ``max_tokens``, ``stop``, ``top_p``.

        Returns:
            The assistant's text content.
        """
        import litellm

        response = litellm.completion(model=model, messages=messages, **kw)
        return _extract_content(response)


@dataclass
class VLLMBackend:
    """In-process vLLM backend.

    The :class:`vllm.LLM` engine is constructed lazily on first use per
    ``model`` id and cached; vLLM model loading is slow (GPU weight
    load), so re-using the engine across calls is the normal pattern.

    Note:
        Requires the ``vllm`` package (``pip install ceng[vllm]``) and a
        CUDA-capable machine.
    """

    name: str = "vllm"
    # ponytail: one engine per model; vllm.LLM build is expensive (loads GPU weights)
    engines: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.engines is None:
            self.engines = {}

    def complete(self, messages: list[dict], model: str, **kw: Any) -> str:
        """Send ``messages`` to a local :class:`vllm.LLM` and return text."""
        import vllm

        from vllm import SamplingParams

        engine = self.engines.get(model)
        if engine is None:
            engine = vllm.LLM(model=model)
            self.engines[model] = engine
        prompt = _messages_to_prompt(messages)
        params = SamplingParams(**_sampling_kwargs(kw))
        outputs = engine.generate([prompt], params)
        return outputs[0].outputs[0].text


@dataclass
class OpenAIBackend:
    """Raw ``openai.OpenAI`` client backend."""

    name: str = "openai"
    client: Any = None

    def __post_init__(self) -> None:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()

    def complete(self, messages: list[dict], model: str, **kw: Any) -> str:
        """Call ``client.chat.completions.create`` and return text."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kw,
        )
        return response.choices[0].message.content


_BACKEND_REGISTRY: dict[str, type[Backend]] = {
    "litellm": LiteLLMBackend,
    "vllm": VLLMBackend,
    "openai": OpenAIBackend,
}

_active_backend: Optional[Backend] = None
_active_lock = threading.Lock()


def available_backends() -> list[str]:
    """Return the names of all built-in backends."""
    return sorted(_BACKEND_REGISTRY.keys())


def get_backend() -> Backend:
    """Return the active backend instance.

    Lazily instantiates the default (``"litellm"``) when none is active.
    """
    global _active_backend
    with _active_lock:
        if _active_backend is None:
            _active_backend = _resolve_backend(None)
        return _active_backend


def set_backend(name: str, **init_kw: Any) -> Backend:
    """Set the active backend by name and return it.

    Args:
        name: One of :func:`available_backends`. Honours the
            ``CENG_BACKEND`` env var when ``name`` is empty.
        **init_kw: Forwarded to the backend constructor.

    Returns:
        The freshly constructed backend (also stored as the active one).

    Raises:
        ValueError: If ``name`` is not a registered backend.
    """
    global _active_backend
    chosen = name or os.environ.get(ENV_BACKEND) or DEFAULT_BACKEND
    with _active_lock:
        backend = _resolve_backend(chosen, **init_kw)
        _active_backend = backend
        return backend


def reset_backend() -> None:
    """Drop the cached backend so the next :func:`get_backend` rebuilds it."""
    global _active_backend
    with _active_lock:
        _active_backend = None


def _resolve_backend(name: Optional[str], **init_kw: Any) -> Backend:
    """Construct a backend by name; ``None`` means default."""
    chosen = name or os.environ.get(ENV_BACKEND) or DEFAULT_BACKEND
    if chosen not in _BACKEND_REGISTRY:
        raise ValueError(
            f"unknown backend {chosen!r}; pick one of {available_backends()}"
        )
    return _BACKEND_REGISTRY[chosen](**init_kw)


def _extract_content(response: Any) -> str:
    """Pull ``choices[0].message.content`` out of an OpenAI-shaped response."""
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError(
            f"could not extract text from LLM response: {response!r}"
        ) from exc


def _messages_to_prompt(messages: list[dict]) -> str:
    """Render a messages list to a single string for chat-tuned vLLM models.

    vLLM's ``LLM.chat`` API exists, but the synchronous ``LLM.generate``
    path is the most widely used across versions; this helper formats
    messages in a conservative templateless way that works for any
    base model.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _sampling_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """Translate OpenAI-style kwargs to vLLM :class:`SamplingParams` fields."""
    mapping = {"temperature": "temperature", "max_tokens": "max_tokens", "top_p": "top_p", "stop": "stop"}
    out: dict[str, Any] = {}
    for src, dst in mapping.items():
        if src in kw:
            out[dst] = kw[src]
    return out
