"""Self-consistency probe from Wolf et al. (2026).

Given a population-level probability question and a tree that
partitions that population, ``ppa_check`` asks the LLM the question
once for the population as a whole and once per leaf of the partition,
then compares the population-level estimate against the prior-weighted
aggregate of the leaf estimates.

If the two estimates diverge by more than ``tolerance``, the model is
guilty of the *macro fallacy*: it gives a worse answer at the
population level than it does when forced to reason about fine-grained
subpopulations first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ceng.backends import Backend, get_backend
from ceng.cache import NAMESPACE_PPA_CHECK, Cache, make_key


PROMPT_VERSION = "1"
POPULATION_PROMPT_TEMPLATE = (
    "Consider the population: {population}\n\n"
    "Question: {question}\n\n"
    "Answer with a single number between 0 and 1 representing the fraction "
    "of the population that answers 'yes'. Return ONLY the number, with no "
    "other text."
)
LEAF_PROMPT_TEMPLATE = (
    "Consider the subpopulation: {description}\n\n"
    "Question: {question}\n\n"
    "Answer with a single number between 0 and 1 representing the fraction "
    "of this subpopulation that answers 'yes'. Return ONLY the number, with "
    "no other text."
)


@dataclass(frozen=True)
class LeafEstimate:
    """One leaf's contribution to the aggregated estimate.

    Attributes:
        description: Human-readable description of the subpopulation.
        prior: Proportion of the overall population represented by this
            subpopulation, in [0, 1]. Aggregated priors across leaves
            of the same parent sum to 1.
        estimate: Probability estimate returned by the LLM for this
            leaf, in [0, 1].
        cache_hit: Whether the estimate came from the cache.
    """

    description: str
    prior: float
    estimate: float
    cache_hit: bool


@dataclass(frozen=True)
class Verdict:
    """Result of a :func:`ppa_check` run.

    Attributes:
        question: The original question.
        population: The population description.
        population_estimate: Estimate returned when asking the question
            directly over the population, in [0, 1].
        aggregated_estimate: Prior-weighted sum of leaf estimates, in
            [0, 1].
        self_consistent: True if the two estimates differ by at most
            ``tolerance``.
        delta: Absolute difference between the two estimates.
        tolerance: Threshold used to decide ``self_consistent``.
        leaves: Per-leaf breakdown, in input order.
        cache_hits: Total number of leaf estimates served from cache.
        cache_misses: Total number of leaf estimates freshly computed.
    """

    question: str
    population: str
    population_estimate: float
    aggregated_estimate: float
    self_consistent: bool
    delta: float
    tolerance: float
    leaves: tuple[LeafEstimate, ...]
    cache_hits: int
    cache_misses: int


def ppa_check(
    question: str,
    population: str,
    tree: list[dict],
    *,
    llm: str = "gpt-4o-mini",
    tolerance: float = 0.05,
    backend: Optional[Backend] = None,
    cache_dir: str = ".ceng_cache",
    **call_kw: Any,
) -> Verdict:
    """Run the partition-prompt-aggregate self-consistency probe.

    Args:
        question: A yes/no probability question, e.g. "What fraction of
            users prefer feature X?".
        population: Description of the full population the question
            applies to.
        tree: List of node dicts partitioning the population. Each node
            has ``"description"`` (str), ``"prior"`` (float, optional,
            default ``1.0``), and ``"children"`` (list of nodes,
            optional). A node with no ``"children"`` is a leaf.
        llm: Model id understood by the active backend.
        tolerance: Maximum acceptable absolute difference between the
            direct population-level estimate and the aggregated
            estimate.
        backend: Optional :class:`Backend`; defaults to the active one.
        cache_dir: Path to the on-disk cache; pass ``""`` to disable.
        **call_kw: Forwarded to ``backend.complete`` for every call
            (``temperature``, ``max_tokens``, etc.).

    Returns:
        A :class:`Verdict` describing both estimates and the per-leaf
        breakdown.

    Raises:
        ValueError: If ``tolerance`` is negative, ``tree`` is empty, or
            any node is malformed.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not tree:
        raise ValueError("tree must be a non-empty list of node dicts")
    backend = backend or get_backend()
    cache = Cache(cache_dir=cache_dir) if cache_dir else _NullCache()
    parsed_tree = _validate_tree(tree)

    population_estimate = _ask_population(
        question=question,
        population=population,
        llm=llm,
        backend=backend,
        cache=cache,
        call_kw=call_kw,
    )
    leaves = _flatten(parsed_tree)
    estimates: list[LeafEstimate] = []
    for description, prior in leaves:
        value, cache_hit = _ask_leaf(
            description=description,
            question=question,
            llm=llm,
            backend=backend,
            cache=cache,
            call_kw=call_kw,
        )
        estimates.append(
            LeafEstimate(description=description, prior=prior, estimate=value, cache_hit=cache_hit)
        )
    aggregated = sum(e.prior * e.estimate for e in estimates)
    delta = abs(population_estimate - aggregated)
    return Verdict(
        question=question,
        population=population,
        population_estimate=population_estimate,
        aggregated_estimate=aggregated,
        self_consistent=delta <= tolerance,
        delta=delta,
        tolerance=tolerance,
        leaves=tuple(estimates),
        cache_hits=sum(1 for e in estimates if e.cache_hit),
        cache_misses=sum(1 for e in estimates if not e.cache_hit),
    )


def _validate_tree(tree: list[dict]) -> list[_TreeNode]:
    """Parse and validate the user-supplied tree; raise on inconsistencies."""
    parsed: list[_TreeNode] = []
    for raw in tree:
        if not isinstance(raw, dict):
            raise ValueError(f"tree node must be a dict, got {type(raw).__name__}")
        description = raw.get("description")
        if not isinstance(description, str) or not description:
            raise ValueError("tree node must have a non-empty 'description'")
        prior = raw.get("prior", 1.0)
        if not isinstance(prior, (int, float)) or not 0 <= prior <= 1:
            raise ValueError(
                f"tree node 'prior' must be between 0 and 1, got {prior!r}"
            )
        children_raw = raw.get("children", [])
        if children_raw:
            children = _validate_tree(children_raw)
            child_total = sum(c.prior for c in children)
            if child_total > 1 + 1e-9:
                raise ValueError(
                    f"children of {description!r} sum to {child_total}; expected <= 1"
                )
        else:
            children = []
        parsed.append(_TreeNode(description=description, prior=float(prior), children=children))
    return parsed


@dataclass
class _TreeNode:
    """Internal tree representation with validation baked in."""

    description: str
    prior: float
    children: list["_TreeNode"] = field(default_factory=list)


def _flatten(nodes: list[_TreeNode], parent_prior: float = 1.0) -> list[tuple[str, float]]:
    """Walk the tree, returning ``(description, effective_prior)`` per leaf.

    Args:
        nodes: Validated tree nodes.
        parent_prior: Cumulative prior inherited from ancestors.

    Returns:
        Pairs of ``(description, prior)`` for every leaf, including
        leaves at depth zero (single-node "trees").
    """
    leaves: list[tuple[str, float]] = []
    for n in nodes:
        effective = parent_prior * n.prior
        if not n.children:
            leaves.append((n.description, effective))
        else:
            leaves.extend(_flatten(n.children, parent_prior=effective))
    return leaves


def _ask_population(
    *,
    question: str,
    population: str,
    llm: str,
    backend: Backend,
    cache: Any,
    call_kw: dict,
) -> float:
    """Ask the population-level question, with cache."""
    key = make_key(
        {
            "op": "ppa_population",
            "model": llm,
            "version": PROMPT_VERSION,
            "question": question,
            "population": population,
        }
    )
    cached = cache.get(NAMESPACE_PPA_CHECK, key) if cache else None
    if cached is not None:
        return float(cached["estimate"])
    prompt = POPULATION_PROMPT_TEMPLATE.format(question=question, population=population)
    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", 8)
    raw = backend.complete(
        messages=[{"role": "user", "content": prompt}],
        model=llm,
        **kw,
    )
    value = _parse_probability(raw)
    if cache:
        cache.set(NAMESPACE_PPA_CHECK, key, {"estimate": value})
    return value


def _ask_leaf(
    *,
    description: str,
    question: str,
    llm: str,
    backend: Backend,
    cache: Any,
    call_kw: dict,
) -> tuple[float, bool]:
    """Ask a leaf's question, with cache. Returns ``(value, cache_hit)``."""
    key = make_key(
        {
            "op": "ppa_leaf",
            "model": llm,
            "version": PROMPT_VERSION,
            "question": question,
            "description": description,
        }
    )
    cached = cache.get(NAMESPACE_PPA_CHECK, key) if cache else None
    if cached is not None:
        return float(cached["estimate"]), True
    prompt = LEAF_PROMPT_TEMPLATE.format(question=question, description=description)
    kw = dict(call_kw)
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", 8)
    raw = backend.complete(
        messages=[{"role": "user", "content": prompt}],
        model=llm,
        **kw,
    )
    value = _parse_probability(raw)
    if cache:
        cache.set(NAMESPACE_PPA_CHECK, key, {"estimate": value})
    return value, False


_PROBABILITY_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _parse_probability(raw: str) -> float:
    """Extract the first number in ``[0, 1]`` from an LLM response.

    Args:
        raw: Text returned by the LLM.

    Returns:
        A float clamped to ``[0, 1]``.

    Raises:
        ValueError: If no number is found.
    """
    if raw is None:
        raise ValueError("LLM returned no content")
    match = _PROBABILITY_PATTERN.search(raw)
    if match is None:
        raise ValueError(f"could not parse probability from LLM response: {raw!r}")
    value = float(match.group(0))
    return max(0.0, min(1.0, value))


class _NullCache:
    """No-op cache used when ``cache_dir`` is empty."""

    def get(self, *_args: Any, **_kw: Any) -> None:
        return None

    def set(self, *_args: Any, **_kw: Any) -> None:
        return None
