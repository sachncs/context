"""DDXPlus (medical reasoning) data processor.

DDXPlus is a 4-way differential-diagnosis benchmark: given a
patient's symptoms, output one of four possible diagnoses. The ACE
paper reports DeepSeek-V3.1+ACE hitting 90.2 vs base 75.2 (+15.0 —
the largest single-benchmark delta in the paper).

The dataset is publicly available on HuggingFace
(`../ddxplus/...`). The processor implemented here works with the
flattened JSONL shape:

    {"id": "...", "question": "...", "options": [...], "answer": "..."}

where ``answer`` is the index (as a string) of the correct option.

This module ships a curated seed playbook with medical-reasoning
strategies so an empty Evolver.run start point is already
domain-aware.
"""

from __future__ import annotations

from ceng.eval import DataSample


class DDXPlusProcessor:
    """Coerce DDXPlus rows to ``DataSample``; grade by exact option match."""

    def process_task_data(self, raw_data: list[dict]) -> list[DataSample]:
        out: list[DataSample] = []
        for row in raw_data:
            question = row.get("question") or row.get("text") or ""
            options = row.get("options") or row.get("choices") or []
            answer = str(row.get("answer") or row.get("label") or "").strip()
            if not question or not options:
                continue
            options_text = "\n".join(
                f"{i}. {opt}" for i, opt in enumerate(options)
            )
            full_question = (
                f"{question}\n\nOptions:\n{options_text}\n\n"
                "Answer with the option number only."
            )
            out.append(
                DataSample(
                    question=full_question,
                    target=answer,
                    context=question,
                    others={
                        "id": row.get("id"),
                        "options": options,
                    },
                )
            )
        return out

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """True iff the first option number extracted matches ground_truth.

        Whitespace-tolerant; ``" 2"``, ``"2."``, and ``"2)"`` all
        parse to 2.
        """
        return _first_int(predicted) == _first_int(ground_truth)

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


def _first_int(text: str) -> int | None:
    import re

    m = re.search(r"[-+]?\d+", text or "")
    return int(m.group(0)) if m else None


def seed_playbook() -> str:
    """Curated medical-reasoning seed playbook for DDXPlus.

    ~25 strategies across the seven ACE sections.
    """
    return _DDX_SEED_PLAYBOOK


_DDX_SEED_PLAYBOOK = """## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: The DDXPlus question lists PATIENT symptoms; the correct answer is the SINGLE most likely diagnosis out of four. Do not list multiple options.
[str-00002] helpful=0 harmful=0 :: Read ALL of the listed symptoms, including 'auxiliary' ones (smoking, alcohol, etc.) — they are the discriminator between similar options.
[str-00003] helpful=0 harmful=0 :: Demographics (age, sex) often pre-filter the option space before symptoms. Apply them first.
[str-00004] helpful=0 harmful=0 :: When two options look similar, look at the SYMPTOM-SPECIFIC features (e.g. fever, blood pressure) that distinguish them.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: Probability scores in the question context (if present) are baseline priors; combine with symptom likelihoods, don't just pick the highest prior.
[cal-00002] helpful=0 harmful=0 :: When computing likelihood ratios, multiply conditional probabilities: P(D | S1, S2) ~ P(D) * P(S1 | D) * P(S2 | D).

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Output format: a single integer (the option index). No words, no explanation, no list.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Don't pick the diagnosis that's most common in the general population — pick the one most consistent with the SPECIFIC symptoms listed.
[mis-00002] helpful=0 harmful=0 :: Don't infer symptoms that aren't in the question. Base the answer on what's there.
[mis-00003] helpful=0 harmful=0 :: Don't output the option name; output the INDEX number. The grader compares numbers, not strings.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: For each option, ask: 'Does every listed symptom match this diagnosis, or are some unrelated?' The option that requires the fewest contortions is usually right.
[ps-00002] helpful=0 harmful=0 :: When in doubt, the option that mentions the most patient symptoms in its description is the stronger candidate.

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: Lab values (blood pressure, heart rate, temperature) are diagnostic discriminators — read them carefully even if they look like noise.
[cc-00002] helpful=0 harmful=0 :: Family history is suggestive, not definitive. Don't overweight it relative to current symptoms.

## Others
[oth-00001] helpful=0 harmful=0 :: When in doubt, output the option that matches the most symptoms in the question. Random guessing has a 25% baseline; deliberate choice should beat that comfortably.
"""
