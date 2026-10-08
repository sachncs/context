"""`Context`: an immutable conversation plus the services to transform it."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from foveate import compression, errors, persistence, verification
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.compression import report as report_lib
from foveate.internals import runner
from foveate.okf import codec as okf_codec  # noqa: F401 - registers "okf"


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Context:
    """An immutable sequence of messages bound to a `Runtime`.

    Every operation returns a new `Context`; nothing is mutated in place.

    Attributes:
        messages: The conversation, oldest first.
        runtime: Backend, cache, tokenizer and observers used by operations.
        report: Provenance of the compression that produced this context,
            or None for a context that was never compressed.
        metadata: Free-form string annotations carried along.
    """

    messages: tuple[messages_lib.Message, ...]
    runtime: runtime_lib.Runtime = dataclasses.field(
        default_factory=runtime_lib.Runtime.from_env
    )
    report: report_lib.CompressionReport | None = None
    metadata: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        coerced = tuple(self.messages)
        for message in coerced:
            if not isinstance(message, messages_lib.Message):
                raise errors.ValidationError(
                    "messages must be Message objects; use Context.from_dicts "
                    f"for mappings (got {type(message).__name__})"
                )
        object.__setattr__(self, "messages", coerced)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @classmethod
    def from_dicts(
        cls,
        messages: Sequence[Mapping[str, object]],
        runtime: runtime_lib.Runtime | None = None,
    ) -> Context:
        """Builds a context from provider-style message mappings.

        Args:
            messages: Mappings with `role` and `content` keys.
            runtime: Runtime to use; defaults to `Runtime.from_env()`.

        Raises:
            ValidationError: If any mapping is malformed.
        """
        parsed = tuple(messages_lib.Message.from_mapping(m) for m in messages)
        if runtime is None:
            return cls(parsed)
        return cls(parsed, runtime)

    def to_dicts(self) -> list[dict[str, str]]:
        """Returns provider-style mappings for every message."""
        return [m.to_mapping() for m in self.messages]

    @property
    def token_count(self) -> int:
        """Returns the summed content tokens (message framing excluded)."""
        tokenizer = self.runtime.tokenizer
        return sum(tokenizer.count(m.content) for m in self.messages)

    async def acompress(
        self,
        method: str | compression.Compressor = "ppa",
        *,
        budget: int | report_lib.Budget,
        **options: object,
    ) -> Context:
        """Compresses this context to fit `budget` tokens.

        Args:
            method: Registered strategy name, a composite such as
                `"ushape+ppa"` (pipeline) or `"ppa|extractive"` (fallback),
                or a configured `Compressor` instance.
            budget: Token ceiling, or a `Budget` with an overflow policy.
            **options: Strategy options; see `compression.resolve`.

        Returns:
            A new context whose `report` describes the run.

        Raises:
            ConfigError: For unknown methods or options.
            CompressionError: If a step fails.
            BudgetExceededError: If the budget is unreachable and the
                overflow policy is `RAISE`.
        """
        strategy = compression.resolve(method, **options)
        return await strategy.compress(self, report_lib.Budget.of(budget))

    def compress(
        self,
        method: str | compression.Compressor = "ppa",
        *,
        budget: int | report_lib.Budget,
        **options: object,
    ) -> Context:
        """Synchronous form of `acompress`; see there for details."""
        return runner.run_sync(self.acompress(method, budget=budget, **options))

    async def averify(
        self,
        method: str | verification.Verifier = "fits",
        **options: object,
    ) -> verification.Verdict:
        """Runs a verification against this context.

        Args:
            method: Registered verifier name (`fits`, `macro_fallacy`) or a
                configured `Verifier`.
            **options: Verifier options; see `verification.resolve`.

        Returns:
            The verifier's verdict (a `Verdict` subclass).

        Raises:
            ConfigError: For unknown methods or invalid options.
            CompressionError: If the verifier's LLM calls fail.
        """
        return await verification.resolve(method, **options).verify(self)

    def verify(
        self,
        method: str | verification.Verifier = "fits",
        **options: object,
    ) -> verification.Verdict:
        """Synchronous form of `averify`; see there for details."""
        return runner.run_sync(self.averify(method, **options))

    def snapshot(self) -> persistence.Snapshot:
        """Returns the persistable parts of this context."""
        return persistence.Snapshot(self.messages, self.report, self.metadata)

    def save(self, path: str | pathlib.Path, format: str = "okf") -> None:
        """Writes this context to `path`.

        Args:
            path: Destination (a directory for `okf`, a file for `json`).
            format: Registered codec name.

        Raises:
            ConfigError: For unknown formats.
        """
        persistence.Codec.registry.get(format)().write(
            self.snapshot(), pathlib.Path(path)
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        format: str = "okf",
        runtime: runtime_lib.Runtime | None = None,
    ) -> Context:
        """Reads a context saved by `save`.

        Args:
            path: Source written by `save`.
            format: Registered codec name.
            runtime: Runtime to attach; defaults to `Runtime.from_env()`.

        Raises:
            ConfigError: For unknown formats.
            ValidationError: If the content is malformed.
        """
        snapshot = persistence.Codec.registry.get(format)().read(
            pathlib.Path(path)
        )
        base = (
            cls(snapshot.messages, runtime)
            if runtime
            else cls(snapshot.messages)
        )
        return dataclasses.replace(
            base, report=snapshot.report, metadata=snapshot.metadata
        )
