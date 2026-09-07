"""Formula (XBRL numeric computation) data processor.

Formula is a financial-calculation benchmark: given a piece of
XBRL-tagged text plus a natural-language question, output a numeric
answer. The ACE paper reports DeepSeek-V3.1+ACE hitting 85.5 vs base
67.5 (+18.0 — the largest single benchmark delta in the paper).

The dataset is publicly available on HuggingFace
(`../finer-xbrl/...` under the formula split). Coerces into
``DataSample`` in :meth:`FormulaProcessor.process_task_data`.

This module ships a curated seed playbook with financial-formula
strategies so an empty Evolver.run start point is already
domain-aware.
"""

from __future__ import annotations

from ceng.eval import DataSample


class FormulaProcessor:
    """Coerce Formula rows to ``DataSample``; grade by numeric match.

    Each Formula row is one (context, question) pair with a numeric
    ground-truth answer. We ask the model for just the number (no
    explanation) and compare with float tolerance.
    """

    float_tolerance: float = 0.01

    def process_task_data(self, raw_data: list[dict]) -> list[DataSample]:
        out: list[DataSample] = []
        for row in raw_data:
            context = row.get("context") or row.get("passage") or ""
            question = (
                row.get("question")
                or row.get("query")
                or row.get("text")
                or ""
            )
            target = str(row.get("answer") or row.get("target") or "").strip()
            out.append(
                DataSample(
                    question=f"Context: {context}\n\nQuestion: {question}",
                    target=target,
                    context=context,
                    others={"id": row.get("id")},
                )
            )
        return out

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return _numbers_close(predicted, ground_truth, self.float_tolerance)

    def evaluate_accuracy(
        self,
        predictions: list[str],
        ground_truths: list[str],
    ) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            1
            for p, g in zip(predictions, ground_truths)
            if self.answer_is_correct(p, g)
        )
        return correct / len(predictions)


def _numbers_close(predicted: str, ground_truth: str, tolerance: float) -> bool:
    """True if a single number can be extracted from both and they're within
    ``tolerance`` of each other (relative)."""
    p = _first_number(predicted)
    g = _first_number(ground_truth)
    if p is None or g is None:
        return False
    if g == 0:
        return abs(p) <= tolerance
    return abs(p - g) / max(abs(g), 1e-9) <= tolerance


def _first_number(text: str) -> float | None:
    import re

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
    return float(m.group(0)) if m else None


def seed_playbook() -> str:
    """Curated financial-formula seed playbook for Formula.

    ~25 strategies across the seven ACE sections.
    """
    return _FORMULA_SEED_PLAYBOOK


_FORMULA_SEED_PLAYBOOK = """## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: When the question asks for a numeric answer, respond with ONLY the number. No commas, no currency symbols, no units.
[str-00002] helpful=0 harmful=0 :: When the question is about a ratio, percentage, or rate, the answer is usually a decimal (e.g. 0.15) or a percent (e.g. 15.0). Read the wording carefully.
[str-00003] helpful=0 harmful=0 :: When the question asks for a sum or total, add the numbers in the relevant period, not the cumulative.
[str-00004] helpful=0 harmful=0 :: Currency amounts in XBRL are usually in raw units (e.g. 1500000 not 1.5M). Convert only if the question explicitly asks.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Net income = revenue - expenses.
[cal-00002] helpful=0 harmful=0 :: EPS = net income / weighted average shares.
[cal-00003] helpful=0 harmful=0 :: ROA = net income / total assets.
[cal-00004] helpful=0 harmful=0 :: Current ratio = current assets / current liabilities.
[cal-00005] helpful=0 harmful=0 :: Year-over-year growth = (this year - last year) / last year.

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: a single number on a line by itself. No words, no $ sign, no comma separators.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Don't return the formula. Return the result of evaluating it.
[mis-00002] helpful=0 harmful=0 :: Don't include units ($ , %, etc.) in the numeric answer.
[mis-00003] helpful=0 harmful=0 :: Don't compute the cumulative figure when the question asks for a single period.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: When the question gives a fiscal year or quarter, restrict the search to that period only.
[ps-00002] helpful=0 harmful=0 :: When the question asks for a ratio and the data is in two different scales, normalise before dividing.

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Words like 'revenue', 'sales', 'turnover' usually point to the top-line figure.
[cc-00002] helpful=0 harmful=0 :: 'Operating income' excludes non-operating items; 'gross income' excludes deductions.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, return a single integer or float with no thousand-separator. The grader uses a numeric tolerance, not a string match.
"""
