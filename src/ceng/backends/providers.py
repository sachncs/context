"""Concrete provider backends. SDKs are imported lazily on first use."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from ceng import errors
from ceng import usage as usage_lib
from ceng.backends import base, classify


def require(module: str, extra: str) -> Any:
    """Imports an optional dependency or raises a helpful `ConfigError`."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise errors.ConfigError(
            f"{module} is required: pip install 'ceng-context[{extra}]'"
        ) from exc


def text_from_choices(response: Any) -> str:
    """Extracts completion text from an OpenAI-shaped response.

    Raises:
        ValidationError: If the response has no text content.
    """
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise errors.ValidationError("response has no choices") from exc
    if not isinstance(text, str) or not text.strip():
        raise errors.ValidationError("response content is empty")
    return text


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
    return kwargs


@base.Backend.registry.register("litellm")
class LiteLLMBackend(base.Backend):
    """Any provider supported by LiteLLM (OpenAI, Anthropic, Bedrock, ...)."""

    async def complete(self, request: base.Request) -> base.Completion:
        litellm = require("litellm", "litellm")
        started = time.monotonic()
        try:
            response = await litellm.acompletion(
                num_retries=0, **payload(request)
            )
        except Exception as exc:
            raise classify.classify(exc) from exc
        return base.Completion(
            text=text_from_choices(response),
            usage=usage_from_response(response),
            model=request.model,
            seconds=time.monotonic() - started,
        )


@base.Backend.registry.register("openai")
class OpenAIBackend(base.Backend):
    """The official OpenAI SDK (also OpenAI-compatible servers)."""

    def __init__(self, base_url: str | None = None) -> None:
        """Creates the backend; the client is built lazily.

        Args:
            base_url: Override for OpenAI-compatible endpoints.
        """
        self.base_url = base_url
        self.client: Any = None

    def get_client(self) -> Any:
        """Builds the async client on first use."""
        if self.client is None:
            openai = require("openai", "openai")
            self.client = openai.AsyncOpenAI(
                base_url=self.base_url, max_retries=0
            )
        return self.client

    async def complete(self, request: base.Request) -> base.Completion:
        client = self.get_client()
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(**payload(request))
        except Exception as exc:
            raise classify.classify(exc) from exc
        return base.Completion(
            text=text_from_choices(response),
            usage=usage_from_response(response),
            model=request.model,
            seconds=time.monotonic() - started,
        )

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


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
        if not text.strip():
            raise errors.ValidationError("vllm returned empty text")
        return base.Completion(
            text=text,
            usage=usage_lib.Usage(
                len(output.prompt_token_ids), len(output.outputs[0].token_ids)
            ),
            model=request.model,
            seconds=time.monotonic() - started,
        )

    async def complete(self, request: base.Request) -> base.Completion:
        return await asyncio.to_thread(self.generate, request)
