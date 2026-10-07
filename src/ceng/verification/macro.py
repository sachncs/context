"""Macro-fallacy self-consistency probe (Wolf et al., 2026).

Asks the model a population-level probability question once directly and
once per leaf of a partition tree, then compares the direct answer with the
prior-weighted aggregate of the leaf answers. A large gap means the model
reasons worse about the whole population than about its parts.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from ceng import errors, prompts
from ceng import messages as messages_lib
from ceng.internals import concurrency
from ceng.verification import base

if TYPE_CHECKING:
    from ceng import context as context_lib

POPULATION = prompts.PromptTemplate(
    name="macro.population",
    version="2",
    system="You answer with a single number and nothing else.",
    user=(
        "Consider the population: {population}\n\nQuestion: {question}\n\n"
        "Answer with a single number between 0 and 1 representing the "
        "fraction of the population that answers 'yes'. Return ONLY the "
        "number, with no other text."
    ),
)
LEAF = prompts.PromptTemplate(
    name="macro.leaf",
    version="2",
    system="You answer with a single number and nothing else.",
    user=(
        "Consider the subpopulation: {description}\n\nQuestion: {question}\n\n"
        "Answer with a single number between 0 and 1 representing the "
        "fraction of this subpopulation that answers 'yes'. Return ONLY the "
        "number, with no other text."
    ),
)
NUMBER = re.compile(r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*(%?)")
SUM_TOLERANCE = 1e-9
ANSWER_MAX_TOKENS = 8


def parse_probability(raw: str) -> float:
    """Extracts the first probability from a model answer.

    A trailing `%` divides by 100 ("50%" is 0.5). Values outside [0, 1]
    without a percent sign are rejected rather than clamped, so a confused
    answer such as "75" fails loudly instead of silently becoming 1.0.

    Raises:
        ValidationError: If no number is found or it is out of range.
    """
    match = NUMBER.search(raw)
    if match is None:
        raise errors.ValidationError(f"no probability in answer: {raw!r}")
    value = float(match.group(1))
    if match.group(2):
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise errors.ValidationError(f"probability out of range: {raw!r}")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class TreeNode:
    """A node of the population partition tree.

    Attributes:
        description: Non-empty description of the subpopulation.
        prior: Share of the parent population, in (0, 1].
        children: Sub-partitions; their priors must sum to exactly 1.
    """

    description: str
    prior: float = 1.0
    children: tuple[TreeNode, ...] = ()

    def __post_init__(self) -> None:
        if not self.description:
            raise errors.ValidationError("tree node needs a description")
        if not 0.0 < self.prior <= 1.0:
            raise errors.ValidationError(
                f"prior of {self.description!r} must be in (0, 1], "
                f"got {self.prior!r}"
            )
        if self.children:
            check_sums(self.children, self.description)

    @classmethod
    def build(cls, raw: Sequence[Mapping[str, object]]) -> tuple[TreeNode, ...]:
        """Builds a validated tree from nested mappings, without recursion.

        Each mapping has `description`, optional `prior` (default 1.0) and
        optional `children`. Iterative construction keeps arbitrarily deep
        trees clear of the interpreter's recursion limit.

        Raises:
            ValidationError: For malformed nodes or priors that do not sum
                to 1 among siblings (including the roots).
        """
        frames: list[Frame] = [Frame(list(raw), None)]
        result: tuple[TreeNode, ...] = ()
        while frames:
            frame = frames[-1]
            if frame.cursor < len(frame.items):
                item = frame.items[frame.cursor]
                frame.cursor += 1
                description, prior = parse_node(item)
                children = item.get("children") or []
                if not isinstance(children, Sequence):
                    raise errors.ValidationError("children must be a list")
                if children:
                    frames.append(Frame(list(children), (description, prior)))
                else:
                    frame.built.append(cls(description, prior))
                continue
            frames.pop()
            nodes = tuple(frame.built)
            if frame.owner is None:
                check_sums(nodes, "the root")
                result = nodes
            else:
                description, prior = frame.owner
                frames[-1].built.append(cls(description, prior, nodes))
        return result

    def leaves(self, ancestor_prior: float = 1.0) -> list[tuple[str, float]]:
        """Returns `(description, effective prior)` for every leaf."""
        found: list[tuple[str, float]] = []
        stack = [(self, ancestor_prior)]
        while stack:
            node, inherited = stack.pop()
            effective = inherited * node.prior
            if not node.children:
                found.append((node.description, effective))
            else:
                stack.extend((c, effective) for c in reversed(node.children))
        return found


@dataclasses.dataclass(slots=True)
class Frame:
    """Work item for the iterative tree builder."""

    items: list[Mapping[str, object]]
    owner: tuple[str, float] | None
    cursor: int = 0
    built: list[TreeNode] = dataclasses.field(default_factory=list)


def parse_node(item: Mapping[str, object]) -> tuple[str, float]:
    """Validates and extracts `(description, prior)` from a raw node."""
    if not isinstance(item, Mapping):
        raise errors.ValidationError(
            f"tree node must be a mapping, got {type(item).__name__}"
        )
    description = item.get("description")
    if not isinstance(description, str) or not description:
        raise errors.ValidationError("tree node needs a non-empty description")
    prior = item.get("prior", 1.0)
    if isinstance(prior, bool) or not isinstance(prior, (int, float)):
        raise errors.ValidationError(
            f"prior of {description!r} is not a number"
        )
    return description, float(prior)


def check_sums(nodes: Sequence[TreeNode], owner: str) -> None:
    """Raises unless sibling priors sum to 1."""
    total = sum(node.prior for node in nodes)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise errors.ValidationError(
            f"children of {owner!r} sum to {total}; expected exactly 1.0"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LeafEstimate:
    """One leaf's contribution.

    Attributes:
        description: Subpopulation description.
        prior: Effective share of the whole population.
        estimate: Model probability for this subpopulation.
        cached: Whether the answer came from the cache.
    """

    description: str
    prior: float
    estimate: float
    cached: bool


@dataclasses.dataclass(frozen=True, slots=True)
class MacroVerdict(base.Verdict):
    """Verdict of the macro-fallacy probe.

    Attributes:
        population_estimate: Direct answer for the whole population.
        aggregated_estimate: Prior-weighted sum of leaf answers.
        delta: Absolute difference of the two.
        tolerance: Largest acceptable delta.
        leaves: Per-leaf breakdown in tree order.
        population_cached: Whether the direct answer came from the cache.
    """

    population_estimate: float = 0.0
    aggregated_estimate: float = 0.0
    delta: float = 0.0
    tolerance: float = 0.0
    leaves: tuple[LeafEstimate, ...] = ()
    population_cached: bool = False


@base.Verifier.register("macro_fallacy")
@dataclasses.dataclass(frozen=True)
class MacroFallacy(base.Verifier):
    """Compares direct and aggregated estimates of a probability.

    Attributes:
        question: Yes/no probability question about the population.
        population: Description of the full population.
        tree: Partition of the population into subpopulations.
        tolerance: Largest acceptable |direct - aggregated|.
    """

    question: str
    population: str
    tree: tuple[TreeNode, ...]
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise errors.ConfigError("tolerance must be non-negative")
        if not self.tree:
            raise errors.ConfigError("tree must contain at least one node")
        if not isinstance(self.tree, tuple):
            raise errors.ConfigError(
                "tree must be a tuple; use TreeNode.build(...)"
            )
        check_sums(self.tree, "the root")

    async def ask(
        self,
        context: context_lib.Context,
        template: prompts.PromptTemplate,
        **values: str,
    ) -> tuple[float, bool]:
        """Asks one probability question; returns `(value, cached)`."""
        messages = (
            messages_lib.Message(messages_lib.Role.SYSTEM, template.system),
            messages_lib.Message(
                messages_lib.Role.USER, template.render_user(**values)
            ),
        )
        completion = await context.runtime.complete(
            messages,
            source=self.name,
            namespace=f"{self.name}:{template.fingerprint}",
            max_tokens=ANSWER_MAX_TOKENS,
        )
        return parse_probability(completion.text), completion.cached

    async def verify(self, context: context_lib.Context) -> MacroVerdict:
        leaves: list[tuple[str, float]] = []
        for root in self.tree:
            leaves.extend(root.leaves())

        def leaf_job(
            description: str,
        ) -> Callable[[], Awaitable[tuple[float, bool]]]:
            async def work() -> tuple[float, bool]:
                return await self.ask(
                    context,
                    LEAF,
                    description=description,
                    question=self.question,
                )

            return work

        try:
            direct, direct_cached = await self.ask(
                context,
                POPULATION,
                population=self.population,
                question=self.question,
            )
            answers = await concurrency.gather_bounded(
                [leaf_job(item[0]) for item in leaves],
                context.runtime.concurrency,
            )
        except (errors.BackendError, errors.ValidationError) as exc:
            raise errors.CompressionError(
                f"verification failed: {exc}", step="macro_fallacy", cause=exc
            ) from exc
        estimates = tuple(
            LeafEstimate(description, prior, value, cached)
            for (description, prior), (value, cached) in zip(
                leaves, answers, strict=True
            )
        )
        aggregated = sum(e.prior * e.estimate for e in estimates)
        delta = abs(direct - aggregated)
        return MacroVerdict(
            verifier=self.name,
            passed=delta <= self.tolerance,
            detail=f"direct={direct:.3f} aggregated={aggregated:.3f}",
            population_estimate=direct,
            aggregated_estimate=aggregated,
            delta=delta,
            tolerance=self.tolerance,
            leaves=estimates,
            population_cached=direct_cached,
        )
