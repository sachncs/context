"""Concrete provider backends. SDKs are imported lazily on first use."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from foveate import errors
from foveate import usage as usage_lib
from foveate.backends import base, classify
from foveate.internals import looplocal


def require(module: str, extra: str) -> Any:
    """Imports an optional dependency or raises a helpful `ConfigError`."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise errors.ConfigError(
            f"{module} is required: pip install 'foveate[{extra}]'"
        ) from exc


def text_from_choices(response: Any) -> tuple[str, str]:
    """Extracts `(text, finish_reason)` from an OpenAI-shaped response.

    Reasoning models can spend the whole token cap on hidden reasoning and
    return `content=None` with `finish_reason="length"`; that is reported as
    empty text so the caller can retry with more headroom.

    Raises:
        ValidationError: If the response has no choices.
    """
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise errors.ValidationError("response has no choices") from exc
    text = content if isinstance(content, str) else ""
    return text, str(getattr(choice, "finish_reason", "") or "")


def usage_from_response(response: Any) -> usage_lib.Usage:
    """Reads token usage from an OpenAI-shaped response (0s if absent)."""
    raw = getattr(response, "usage", None)
    return usage_lib.Usage(
        int(getattr(raw, "prompt_tokens", 0) or 0),
        int(getattr(raw, "completion_tokens", 0) or 0),
    )


def payload(request: base.Request) -> dict[str, Any]:
    """Builds OpenAI-style keyword arguments for `request`."""
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": [m.to_mapping() for m in request.messages],
        "temperature": request.temperature,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.options:
        kwargs["extra_body"] = dict(request.options)
    return kwargs


def completion_from_response(
    response: Any, request: base.Request, started: float
) -> base.Completion:
    """Builds a `Completion` from an OpenAI-shaped response."""
    text, finish_reason = text_from_choices(response)
    return base.Completion(
        text=text,
        usage=usage_from_response(response),
        model=request.model,
        seconds=time.monotonic() - started,
        finish_reason=finish_reason,
    )


@base.Backend.registry.register("litellm")
class LiteLLMBackend(base.Backend):
    """Any provider supported by LiteLLM (OpenAI, Anthropic, Bedrock, ...)."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None
    ) -> None:
        """Stores optional endpoint overrides.

        Args:
            base_url: Override for the provider endpoint (`api_base`).
            api_key: Explicit key; otherwise LiteLLM reads the provider's
                usual environment variable.
        """
        self.base_url = base_url
        self.api_key = api_key

    async def complete(self, request: base.Request) -> base.Completion:
        litellm = require("litellm", "litellm")
        started = time.monotonic()
        kwargs = payload(request)
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            response = await litellm.acompletion(num_retries=0, **kwargs)
        except Exception as exc:
            raise classify.classify(exc) from exc
        return completion_from_response(response, request, started)


@base.Backend.registry.register("openai")
class OpenAIBackend(base.Backend):
    """The official OpenAI SDK (also OpenAI-compatible servers)."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None
    ) -> None:
        """Creates the backend; the client is built lazily.

        Args:
            base_url: Override for OpenAI-compatible endpoints.
            api_key: Explicit key; otherwise the SDK reads `OPENAI_API_KEY`.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.clients: looplocal.LoopLocal[Any] = looplocal.LoopLocal(
            self.build_client
        )

    def build_client(self) -> Any:
        """Builds an async client (called once per event loop)."""
        openai = require("openai", "openai")
        return openai.AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, max_retries=0
        )

    async def complete(self, request: base.Request) -> base.Completion:
        client = self.clients.get()
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(**payload(request))
        except Exception as exc:
            raise classify.classify(exc) from exc
        return completion_from_response(response, request, started)

    async def aclose(self) -> None:
        client = self.clients.discard()
        if client is not None:
            await client.close()


@base.Backend.registry.register("vllm")
class VLLMBackend(base.Backend):
    """In-process vLLM engine. One engine per model, serialized by a lock."""

    def __init__(self, **engine_kwargs: Any) -> None:
        """Stores engine options; the engine loads on first call."""
        self.engine_kwargs = engine_kwargs
        self.engines: dict[str, Any] = {}
        self.lock = threading.Lock()

    def generate(self, request: base.Request) -> base.Completion:
        """Runs a blocking generation; call via a worker thread."""
        vllm = require("vllm", "vllm")
        started = time.monotonic()
        with self.lock:
            if request.model not in self.engines:
                self.engines[request.model] = vllm.LLM(
                    model=request.model, **self.engine_kwargs
                )
            engine = self.engines[request.model]
            params = vllm.SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens or 512,
            )
            try:
                outputs = engine.chat(
                    [[m.to_mapping() for m in request.messages]], params
                )
            except Exception as exc:
                raise classify.classify(exc) from exc
        output = outputs[0]
        text = output.outputs[0].text
        return base.Completion(
            text=text,
            finish_reason=str(getattr(output.outputs[0], "finish_reason", "")),
            usage=usage_lib.Usage(
                len(output.prompt_token_ids), len(output.outputs[0].token_ids)
            ),
            model=request.model,
            seconds=time.monotonic() - started,
        )

    async def complete(self, request: base.Request) -> base.Completion:
        return await asyncio.to_thread(self.generate, request)
