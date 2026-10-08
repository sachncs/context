"""Scoring: correctness, citation quality, abstention, reliability."""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from collections.abc import Sequence

from foveate import errors, messages, prompts
from foveate import runtime as runtime_lib
from foveate.bench.longdoc import gold, pipelines
from foveate.internals import jsonout

NUMBER = re.compile(r"(?<![\w.])-?\(?\$?-?\d[\d,]*(?:\.\d+)?\)?")
SCALE_WORDS = re.compile(r"\b(million|billion|thousand|trillion|bn|mm)\b", re.I)
SCALES = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
REL_TOLERANCE = 0.01
ABS_TOLERANCE = 0.006
YEAR = range(1900, 2101)

JUDGE = prompts.PromptTemplate(
    name="longdoc.judge",
    version="1",
    system="You are a strict grader. Reply with JSON only.",
    user=(
        "Grade a candidate answer to a question about a company filing.\n"
        "Question: {question}\nReference answer: {expected}\n"
        "Candidate answer: {candidate}\n\n"
        "The candidate is correct if it states the same fact or figure as "
        "the reference. Rounding, formatting, units (millions vs billions) "
        "and extra explanation are fine; a different figure, a wrong "
        "period/company, a refusal, or a missing answer is incorrect.\n"
        'Reply {{"correct": true}} or {{"correct": false}}.'
    ),
)


def parse_numbers(text: str) -> list[float]:
    """Extracts numeric values; parentheses and a leading minus are negative."""
    values = []
    for raw in NUMBER.findall(text):
        negative = raw.startswith("(") or "-" in raw
        digits = re.sub(r"[^0-9.]", "", raw)
        if not digits or digits == ".":
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        values.append(-value if negative else value)
    return values


def is_year(value: float) -> bool:
    """Returns whether `value` looks like a calendar year."""
    return value.is_integer() and int(value) in YEAR


def primary(values: Sequence[float]) -> float | None:
    """Returns the first non-year number (or the first number)."""
    for value in values:
        if not is_year(value):
            return value
    return values[0] if values else None


def close(a: float, b: float) -> bool:
    """Returns whether two figures agree within rounding tolerance."""
    return abs(a - b) <= max(ABS_TOLERANCE, REL_TOLERANCE * abs(b))


def numeric_match(expected: str, predicted: str) -> bool | None:
    """Compares the reference figure with the candidate's figures.

    Returns:
        True on a match, False on a mismatch, None if the reference has no
        number (then a judge must decide). A match up to a power-of-1000
        scale counts only when the candidate states a scale word.
    """
    want = primary(parse_numbers(expected))
    if want is None:
        return None
    candidates = [v for v in parse_numbers(predicted) if not is_year(v)]
    if any(close(v, want) for v in candidates):
        return True
    if SCALE_WORDS.search(predicted):
        return any(close(v * s, want) for v in candidates for s in SCALES)
    return False


async def judge(
    runtime: runtime_lib.Runtime, question: str, expected: str, candidate: str
) -> bool:
    """Asks the judge model whether `candidate` matches `expected`."""
    try:
        reply = await runtime.complete(
            (
                messages.Message(messages.Role.SYSTEM, JUDGE.system),
                messages.Message(
                    messages.Role.USER,
                    JUDGE.render_user(
                        question=question,
                        expected=expected,
                        candidate=candidate,
                    ),
                ),
            ),
            source="judge",
            namespace=f"judge:{JUDGE.fingerprint}",
            max_tokens=400,
        )
        return bool(jsonout.extract_json(reply.text).get("correct"))
    except (errors.BackendError, errors.ValidationError):
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """One scored (item, pipeline, run).

    Attributes:
        item: The gold item.
        pipeline: Pipeline name.
        run: Run index (0 for the deterministic run).
        result: What the pipeline returned.
        correct: Answer correctness (answerable/paraphrase/needle kinds) or
            correct abstention (unanswerable kind).
        method: How correctness was decided (`numeric`, `judge`, `needle`,
            `abstain`, `none`).
        precision: Share of cited pages that are gold pages.
        recall: Share of gold pages that were cited.
    """

    item: gold.GoldItem
    pipeline: str
    run: int
    result: pipelines.Result
    correct: bool
    method: str
    precision: float = 0.0
    recall: float = 0.0


def page_overlap(
    cited: Sequence[int], gold_pages: Sequence[int]
) -> tuple[float, float]:
    """Returns `(precision, recall)` of cited pages against gold pages."""
    if not cited or not gold_pages:
        return 0.0, 0.0
    hit = len(set(cited) & set(gold_pages))
    return hit / len(set(cited)), hit / len(set(gold_pages))


async def score(
    judge_runtime: runtime_lib.Runtime,
    item: gold.GoldItem,
    pipeline: str,
    run: int,
    result: pipelines.Result,
) -> Outcome:
    """Scores one pipeline result against its gold item."""
    if result.error:
        return Outcome(item, pipeline, run, result, False, "error")
    precision, recall = page_overlap(result.cited_pages, item.gold_pages)
    if item.kind == gold.UNANSWERABLE:
        return Outcome(
            item,
            pipeline,
            run,
            result,
            result.abstained,
            "abstain",
            precision,
            recall,
        )
    if result.abstained:
        return Outcome(
            item, pipeline, run, result, False, "abstain", precision, recall
        )
    if item.kind == gold.NEEDLE:
        right = item.expected.casefold() in result.text.casefold()
        return Outcome(
            item, pipeline, run, result, right, "needle", precision, recall
        )
    verdict = numeric_match(item.expected, result.text)
    if verdict is True:
        return Outcome(
            item, pipeline, run, result, True, "numeric", precision, recall
        )
    right = await judge(
        judge_runtime, item.question, item.expected, result.text
    )
    return Outcome(
        item, pipeline, run, result, right, "judge", precision, recall
    )


def mean(values: Sequence[float]) -> float:
    """Returns the mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def agreement(outcomes: Sequence[Outcome]) -> float:
    """Share of source questions whose outcomes all agree on correctness.

    Outcomes are grouped by the original question (`meta["source"]`, else
    the item id), so paraphrases and repeated runs of one question are
    compared with each other.
    """
    groups: dict[str, set[bool]] = defaultdict(set)
    sizes: dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        key = outcome.item.meta.get("source", outcome.item.id)
        groups[key].add(outcome.correct)
        sizes[key] += 1
    multi = [k for k, n in sizes.items() if n > 1]
    return mean([1.0 if len(groups[k]) == 1 else 0.0 for k in multi])
