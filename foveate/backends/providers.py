"""Concrete backends: OpenAI-compatible HTTP (stdlib) and in-process vLLM."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

from foveate import errors
from foveate import usage as usage_lib
from foveate.backends import base, classify, http

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def require(module: str, extra: str) -> Any:
    """Imports an optional dependency or raises a helpful `ConfigError`."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise errors.ConfigError(
            f"{module} is required: pip install 'foveate[{extra}]'"
        ) from exc


def text_from_choices(response: Mapping[str, Any]) -> tuple[str, str]:
    """Extracts `(text, finish_reason)` from an OpenAI-shaped reply.

    Reasoning models can spend the whole token cap on hidden reasoning and
    return `content=null` with `finish_reason="length"`; that is reported as
    empty text so the caller can retry with more headroom.

    Raises:
        ValidationError: If the reply has no choices.
    """
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise errors.ValidationError("response has no choices") from exc
    text = content if isinstance(content, str) else ""
    return text, str(choice.get("finish_reason") or "")


def usage_from_response(response: Mapping[str, Any]) -> usage_lib.Usage:
    """Reads token usage from an OpenAI-shaped reply (0s if absent)."""
    raw = response.get("usage") or {}
    details = raw.get("prompt_tokens_details") or {}
    return usage_lib.Usage(
        int(raw.get("prompt_tokens") or 0),
        int(raw.get("completion_tokens") or 0),
        int(details.get("cached_tokens") or 0),
    )


def payload(request: base.Request) -> dict[str, Any]:
    """Builds the JSON body of a chat-completions call for `request`.

    Provider options (for example `reasoning_effort`) are merged at the top
    level of the body.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [m.to_mapping() for m in request.messages],
        "temperature": request.temperature,
    }
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    body.update(request.options or {})
    return body


def completion_from_response(
    response: Mapping[str, Any], request: base.Request, started: float
) -> base.Completion:
    """Builds a `Completion` from an OpenAI-shaped reply."""
    text, finish_reason = text_from_choices(response)
    return base.Completion(
        text=text,
        usage=usage_from_response(response),
        model=request.model,
        seconds=time.monotonic() - started,
        finish_reason=finish_reason,
    )


@base.Backend.registry.register("openai")
class OpenAIBackend(base.Backend):
    """Any OpenAI-compatible chat-completions endpoint (no SDK needed).

    Works with OpenAI, vLLM, Ollama, NVIDIA, Together, Azure OpenAI gateways
    and similar servers.

    Attributes:
        base_url: Endpoint root, for example `https://api.openai.com/v1`.
        api_key: Bearer token; falls back to `OPENAI_API_KEY`.
        timeout: Seconds before one call is abandoned.
        headers: Extra request headers.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Creates the backend.

        Args:
            base_url: Endpoint root; defaults to `OPENAI_BASE_URL` or OpenAI.
            api_key: Explicit key; otherwise `OPENAI_API_KEY`.
            timeout: Per-call timeout in seconds.
            headers: Extra request headers.
        """
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.headers = dict(headers or {})

    def request_headers(self) -> dict[str, str]:
        """Returns the headers sent with every call."""
        headers = dict(self.headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(self, request: base.Request) -> base.Completion:
        started = time.monotonic()
        response = await http.apost_json(
            f"{self.base_url}/chat/completions",
            payload(request),
            self.request_headers(),
            self.timeout,
        )
        return completion_from_response(response, request, started)


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
