"""Runtime: the explicit, injectable bundle of services foveate operates with.

Replaces process-global singletons. Every LLM call in the library flows
through `Runtime.complete`, which is the single implementation of
cache-lookup, single-flight de-duplication, response validation, usage
accounting and event emission.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import TracebackType

from foveate import errors, models, observability
from foveate import messages as messages_lib
from foveate import tokenizers as tokenizers_lib
from foveate import usage as usage_lib
from foveate.backends import base as backend_base
from foveate.backends import embeddings as embeddings_lib
from foveate.backends import none as none_backend
from foveate.backends import resilient
from foveate.cache import base as cache_base
from foveate.cache import sqlite as cache_sqlite
from foveate.internals import hashing, reasoning, runner

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CACHE_DIR = ".foveate/cache"
VALIDATION_RETRIES = 1
LENGTH_RETRIES = 3
LENGTH_GROWTH = 4
WINDOW_MARGIN = 64


class SingleFlight:
    """Collapses concurrent identical work into one execution.

    If the caller that started the work is cancelled, waiting callers are
    not cancelled with it: one of them takes over and runs the work itself.
    """

    def __init__(self) -> None:
        self.flights: dict[tuple[int, str], asyncio.Future[str]] = {}

    async def run(self, key: str, work: Callable[[], Awaitable[str]]) -> str:
        """Runs `work` once per concurrent `key`; others await its result."""
        flight_key = (id(asyncio.get_running_loop()), key)
        while True:
            existing = self.flights.get(flight_key)
            if existing is None:
                return await self.lead(flight_key, work)
            try:
                return await asyncio.shield(existing)
            except asyncio.CancelledError:
                if not existing.cancelled():
                    raise  # this caller was cancelled, not the leader
                # The leader was cancelled: loop and take over.

    async def lead(
        self,
        flight_key: tuple[int, str],
        work: Callable[[], Awaitable[str]],
    ) -> str:
        """Executes `work` and publishes the outcome to followers."""
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.flights[flight_key] = future
        try:
            result = await work()
        except asyncio.CancelledError:
            future.cancel()
            raise
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved so no "never retrieved" noise
            raise
        else:
            future.set_result(result)
            return result
        finally:
            self.flights.pop(flight_key, None)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Runtime:
    """Services shared by every operation on a `Context`.

    Attributes:
        backend: LLM provider (typically a `ResilientBackend`).
        model: Default model name for requests.
        cache: Completion cache.
        tokenizer: Token counter used for budgeting.
        observers: Event sinks.
        prices: Price table used to compute cost in reports.
        embedder: Embedding model for semantic retrieval, or None.
        context_window: Model window in tokens; None looks it up by name.
        options: Provider parameters sent with every request, e.g.
            `{"reasoning_effort": "low"}` for reasoning models.
        concurrency: Maximum parallel LLM calls issued by one operation.
        single_flight: De-duplicator for concurrent identical requests.
    """

    backend: backend_base.Backend
    model: str = DEFAULT_MODEL
    cache: cache_base.Cache = dataclasses.field(
        default_factory=cache_base.MemoryCache
    )
    tokenizer: tokenizers_lib.Tokenizer = dataclasses.field(
        default_factory=tokenizers_lib.default_tokenizer
    )
    observers: tuple[observability.Observer, ...] = ()
    prices: usage_lib.PriceTable = dataclasses.field(
        default_factory=usage_lib.PriceTable
    )
    options: Mapping[str, object] = dataclasses.field(default_factory=dict)
    embedder: embeddings_lib.Embedder | None = None
    context_window: int | None = None
    concurrency: int = 8
    single_flight: SingleFlight = dataclasses.field(
        default_factory=SingleFlight
    )

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise errors.ConfigError("concurrency must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Runtime:
        """Builds a runtime from `FOVEATE_*` environment variables.

        Recognised: `FOVEATE_BACKEND` (openai|vllm), `FOVEATE_MODEL`,
        `FOVEATE_BASE_URL` (OpenAI-compatible endpoint), `FOVEATE_OPTIONS` (JSON
        object of provider parameters), `FOVEATE_CACHE_DIR` (empty disables
        caching), `FOVEATE_TIMEOUT_SECONDS`,
        `FOVEATE_RETRY_ATTEMPTS`, `FOVEATE_CONCURRENCY`,
        `FOVEATE_RATE_LIMIT_PER_SECOND` (optional) and
        `FOVEATE_DEADLINE_SECONDS` (optional total retry budget).

        Args:
            environ: Mapping to read instead of `os.environ`.

        Raises:
            ConfigError: For unknown backends or unparseable numbers.
        """
        env = os.environ if environ is None else environ

        def number(name: str, default: float, kind: type) -> float:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                return kind(raw)  # type: ignore[no-any-return]
            except ValueError as exc:
                raise errors.ConfigError(
                    f"{name}={raw!r} is not valid"
                ) from exc

        backend_name = env.get("FOVEATE_BACKEND", "openai")
        backend_cls = backend_base.Backend.registry.get(backend_name)
        base_url = env.get("FOVEATE_BASE_URL")
        try:
            inner = (
                backend_cls(base_url=base_url) if base_url else backend_cls()
            )
        except TypeError as exc:
            raise errors.ConfigError(
                f"backend {backend_name!r} does not support FOVEATE_BASE_URL"
            ) from exc
        raw_options = env.get("FOVEATE_OPTIONS", "")
        try:
            options = json.loads(raw_options) if raw_options else {}
        except ValueError as exc:
            raise errors.ConfigError(
                "FOVEATE_OPTIONS is not valid JSON"
            ) from exc
        if not isinstance(options, dict):
            raise errors.ConfigError("FOVEATE_OPTIONS must be a JSON object")
        concurrency = int(number("FOVEATE_CONCURRENCY", 8, int))
        rate = env.get("FOVEATE_RATE_LIMIT_PER_SECOND")
        deadline = env.get("FOVEATE_DEADLINE_SECONDS")
        backend = resilient.ResilientBackend(
            inner,
            rate_per_second=float(
                number("FOVEATE_RATE_LIMIT_PER_SECOND", 0.0, float)
            )
            if rate
            else None,
            deadline=float(number("FOVEATE_DEADLINE_SECONDS", 0.0, float))
            if deadline
            else None,
            retry=resilient.RetryPolicy(
                attempts=int(number("FOVEATE_RETRY_ATTEMPTS", 3, int))
            ),
            timeout=float(number("FOVEATE_TIMEOUT_SECONDS", 60.0, float)),
            max_concurrency=concurrency,
        )
        cache_dir = env.get("FOVEATE_CACHE_DIR", DEFAULT_CACHE_DIR)
        cache: cache_base.Cache = (
            cache_sqlite.SqliteCache(cache_dir)
            if cache_dir
            else cache_base.NullCache()
        )
        model = env.get("FOVEATE_MODEL", DEFAULT_MODEL)
        window_raw = env.get("FOVEATE_CONTEXT_WINDOW")
        try:
            window = int(window_raw) if window_raw else None
        except ValueError as exc:
            raise errors.ConfigError(
                f"FOVEATE_CONTEXT_WINDOW={window_raw!r} is not an integer"
            ) from exc
        embedding_model = env.get("FOVEATE_EMBEDDING_MODEL")
        embedder = (
            embeddings_lib.OpenAIEmbedder(embedding_model, base_url=base_url)
            if embedding_model
            else None
        )
        return cls(
            backend=backend,
            model=model,
            cache=cache,
            tokenizer=tokenizers_lib.for_model(model),
            options=options,
            embedder=embedder,
            context_window=window,
            concurrency=concurrency,
        )

    @classmethod
    def without_llm(cls) -> Runtime:
        """Returns a runtime that can only run LLM-free compression methods.

        Any attempt to call a model fails with `PermanentBackendError`, and
        nothing is cached.
        """
        return cls(
            backend=none_backend.NoBackend(),
            cache=cache_base.NullCache(),
            tokenizer=tokenizers_lib.default_tokenizer(),
        )

    def emit(self, event: observability.Event) -> None:
        """Delivers an event to this runtime's observers."""
        observability.emit(self.observers, event)

    async def complete(
        self,
        messages: Sequence[messages_lib.Message],
        *,
        source: str,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> backend_base.Completion:
        """Completes `messages` through cache, dedup and the backend.

        Args:
            messages: Conversation to complete.
            source: Component name for emitted events.
            namespace: Cache namespace; include the prompt fingerprint and
                strategy version so changes never hit stale entries.
            max_tokens: Completion cap.
            temperature: Sampling temperature.
            model: Model override (defaults to `self.model`).

        Returns:
            The completion (`cached=True` when served from the cache).

        Raises:
            BackendError: If the backend fails after its retry policy.
            ValidationError: If the model returns empty text repeatedly.
        """
        request = backend_base.Request(
            model=model or self.model,
            messages=tuple(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            options=tuple(sorted(self.options.items())),
        )
        key = hashing.fingerprint(namespace, request.fingerprint)
        cached = self.cache.get(key)
        if cached is not None:
            try:
                data = json.loads(cached)
                self.emit(observability.CacheLookup(source=source, hit=True))
                return backend_base.Completion(
                    text=data["text"],
                    usage=usage_lib.Usage(
                        data.get("prompt", 0), data.get("completion", 0)
                    ),
                    model=request.model,
                    cached=True,
                )
            except (ValueError, KeyError, TypeError):
                self.cache.delete(key)
        self.emit(observability.CacheLookup(source=source, hit=False))

        holder: list[backend_base.Completion] = []

        async def work() -> str:
            completion = await self.call_validated(request, source)
            holder.append(completion)
            self.cache.set(
                key,
                json.dumps(
                    {
                        "text": completion.text,
                        "prompt": completion.usage.prompt_tokens,
                        "completion": completion.usage.completion_tokens,
                    }
                ),
            )
            return completion.text

        text = await self.single_flight.run(key, work)
        if holder:
            return holder[0]
        return backend_base.Completion(
            text=text, model=request.model, cached=True
        )

    async def call_validated(
        self, request: backend_base.Request, source: str
    ) -> backend_base.Completion:
        """Calls the backend, recovering from empty or truncated output.

        Two situations are retried, each a bounded number of times:

        * The model hit the token cap before producing (enough) visible
          text, which is typical for reasoning models whose hidden thinking
          consumes the cap. "Not enough" means less than half the cap is
          visible. The cap is multiplied by `LENGTH_GROWTH` for the next
          attempt (`LENGTH_RETRIES` times); a still-truncated reply is then
          returned as is.
        * The model returned empty text for another reason; it is asked
          again `VALIDATION_RETRIES` times.

        Raises:
            ValidationError: If the model still returns no text.
        """
        current = request
        length_left = LENGTH_RETRIES
        empty_left = VALIDATION_RETRIES
        while True:
            started = time.monotonic()
            completion = await self.backend.complete(current)
            completion = dataclasses.replace(
                completion, text=reasoning.strip_reasoning(completion.text)
            )
            self.emit(
                observability.BackendCall(
                    source=source,
                    model=current.model,
                    usage=completion.usage,
                    seconds=time.monotonic() - started,
                )
            )
            visible = self.tokenizer.count(completion.text)
            starved = (
                completion.truncated
                and current.max_tokens is not None
                and visible < current.max_tokens // 2
            )
            if completion.text.strip() and not (starved and length_left > 0):
                return completion
            if (
                completion.truncated
                and length_left > 0
                and current.max_tokens is not None
            ):
                length_left -= 1
                grown = self.cap_to_window(
                    current, current.max_tokens * LENGTH_GROWTH
                )
                if grown <= current.max_tokens:
                    raise errors.ValidationError(
                        "model returned no text (token cap reached and the "
                        "context window leaves no room to grow it)"
                    )
                current = dataclasses.replace(current, max_tokens=grown)
                continue
            if empty_left > 0 and not completion.truncated:
                empty_left -= 1
                continue
            raise errors.ValidationError(
                "model returned no text"
                + (" (token cap reached)" if completion.truncated else "")
            )

    def cap_to_window(self, request: backend_base.Request, wanted: int) -> int:
        """Limits a larger completion cap so prompt plus cap fit the window.

        Without a known `context_window` the wanted cap is returned.
        """
        window = self.context_window
        if window is None:
            info = models.lookup(request.model)
            window = info.context_window if info else None
        if window is None:
            return wanted
        prompt = sum(self.tokenizer.count(m.content) for m in request.messages)
        return max(0, min(wanted, window - prompt - WINDOW_MARGIN))

    async def aclose(self) -> None:
        """Closes the backend and cache."""
        await self.backend.aclose()
        if self.embedder is not None:
            await self.embedder.aclose()
        self.cache.close()

    def close(self) -> None:
        """Synchronous variant of `aclose`."""
        runner.run_sync(self.aclose())

    def __enter__(self) -> Runtime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
