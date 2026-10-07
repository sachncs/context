"""The `Compressor` template and its registry."""

from __future__ import annotations

import abc
import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from ceng import errors, observability, prompts
from ceng import messages as messages_lib
from ceng import usage as usage_lib
from ceng.compression import fit, report
from ceng.internals import registry

if TYPE_CHECKING:
    from ceng import context as context_lib


@dataclasses.dataclass(frozen=True)
class Compressor(abc.ABC):
    """A strategy that reduces a context to fit a token budget.

    Subclasses are frozen dataclasses whose fields are their options, so
    options are validated, hashable and visible in `repr`. Implement `run`;
    the base class supplies the common contract through `compress`:

    1. If the context already fits, return it unchanged (no LLM calls).
    2. Otherwise call `run`.
    3. Enforce the budget: raise `BudgetExceededError` or hard-truncate,
       per `Budget.overflow`.
    4. Attach a `CompressionReport`.
    """

    registry: ClassVar[registry.Registry[type[Compressor]]] = registry.Registry(
        "compression method"
    )
    name: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[Compressor]], type[Compressor]]:
        """Returns a class decorator registering a strategy under `name`.

        The decorator also sets the strategy's `name` class attribute.

        Raises:
            ConfigError: At decoration time if `name` is taken.
        """

        def decorator(subclass: type[Compressor]) -> type[Compressor]:
            cls.registry.add(name, subclass)
            subclass.name = name
            return subclass

        return decorator

    @property
    def label(self) -> str:
        """Returns the method label recorded in reports."""
        return self.name

    @abc.abstractmethod
    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        """Returns a reduced context.

        The result need not fit exactly; `compress` enforces the budget.
        """

    async def compress(
        self, context: context_lib.Context, budget: report.Budget
    ) -> context_lib.Context:
        """Compresses `context` to `budget` and attaches a report.

        Raises:
            CompressionError: If a step fails.
            BudgetExceededError: If the budget is unreachable and the
                overflow policy is `RAISE`.
        """
        started = time.monotonic()
        original = context.token_count
        trace = report.Trace()
        result = context
        if original > budget.tokens:
            result = await self.run(context, budget, trace)
            result = self.enforce(result, budget, trace)
        final = result.token_count
        seconds = time.monotonic() - started
        usage_total = sum(
            (record.usage for record in trace.records),
            start=usage_lib.Usage(),
        )
        cost = context.runtime.prices.cost(context.runtime.model, usage_total)
        context.runtime.emit(
            observability.StepFinished(
                source=self.name, step="compress", seconds=seconds
            )
        )
        return dataclasses.replace(
            result,
            report=report.CompressionReport(
                method=self.label,
                version=self.version,
                budget=budget.tokens,
                original_tokens=original,
                final_tokens=final,
                steps=tuple(trace.records),
                cost_usd=cost,
                truncated=trace.truncated,
                seconds=seconds,
            ),
        )

    def enforce(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        """Applies the budget's overflow policy to an over-budget result."""
        actual = context.token_count
        if actual <= budget.tokens:
            return context
        if budget.overflow is report.Overflow.RAISE:
            raise errors.BudgetExceededError(budget.tokens, actual)
        fitted = fit.fit(
            context.messages, budget.tokens, context.runtime.tokenizer
        )
        result = dataclasses.replace(context, messages=fitted)
        if result.token_count > budget.tokens:
            raise errors.BudgetExceededError(budget.tokens, result.token_count)
        trace.truncated = True
        return result

    async def ask(
        self,
        context: context_lib.Context,
        trace: report.Trace,
        *,
        step: str,
        template: prompts.PromptTemplate,
        max_tokens: int,
        input_tokens: int,
        **values: str,
    ) -> str:
        """Asks the LLM using `template`, recording a step.

        The cache namespace combines the strategy name and version with the
        template fingerprint, so editing either invalidates cached answers.

        Args:
            context: Supplies the runtime.
            trace: Receives the step record.
            step: Label for the record and error messages.
            template: Prompt to render.
            max_tokens: Completion cap.
            input_tokens: Tokens being compressed (for the record).
            **values: Template placeholder values.

        Returns:
            The stripped completion text.

        Raises:
            CompressionError: On backend or validation failure.
        """
        messages = (
            messages_lib.Message(messages_lib.Role.SYSTEM, template.system),
            messages_lib.Message(
                messages_lib.Role.USER, template.render_user(**values)
            ),
        )
        namespace = f"{self.name}:{self.version}:{template.fingerprint}"
        started = time.monotonic()
        try:
            completion = await context.runtime.complete(
                messages,
                source=self.name,
                namespace=namespace,
                max_tokens=max_tokens,
            )
        except (errors.BackendError, errors.ValidationError) as exc:
            raise errors.CompressionError(
                f"{self.name} {step} failed: {exc}", step=step, cause=exc
            ) from exc
        text = completion.text.strip()
        trace.add(
            report.StepRecord(
                name=step,
                input_tokens=input_tokens,
                output_tokens=context.runtime.tokenizer.count(text),
                cached=completion.cached,
                usage=completion.usage,
                seconds=time.monotonic() - started,
            )
        )
        return text
