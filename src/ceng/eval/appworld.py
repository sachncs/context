"""AppWorld data processor stub.

AppWorld is a multi-turn agent benchmark: the model receives a task
description and a list of available apps, and must interact with a
mock environment to complete the task. The official dataset is
gated behind the AppWorld benchmark API (registration at
``appworld.dev`` required); we ship a processor stub plus
documentation here so callers who have credentials can plug in
their own AppWorld samples.

Expected sample shape (one dict per task, one or more steps per task)::

    {
        "id": "...",
        "task_description": "...",
        "available_apps": ["email", "file_system", ...],
        "steps": [
            {"observation": "...", "action": "...",
             "expected_response": "...", "is_correct": True/False},
            ...
        ]
    }

To run a real AppWorld benchmark you need the
``appworld`` Python package (``pip install appworld``) and an
``ALFW_API_KEY`` environment variable. ceng does not depend on
either — they're required only when you actually call
``bench_appworld()``.
"""

from __future__ import annotations

import os

from ceng.eval import DataSample


class AppWorldProcessor:
    """Coerce AppWorld rows to DataSample; grade by step-level success.

    Each step in the AppWorld trace becomes one DataSample. The
    target is ``"1"`` if the step is correct, ``"0"`` otherwise.
    The model is asked to predict the next action; we compare its
    top-line answer against ``expected_response``.
    """

    def process_task_data(self, raw_data: list[dict]) -> list[DataSample]:
        out: list[DataSample] = []
        for task in raw_data:
            task_id = task.get("id", "")
            available = task.get("available_apps", [])
            for i, step in enumerate(task.get("steps", [])):
                observation = step.get("observation", "")
                expected = step.get("expected_response", "")
                is_correct = step.get("is_correct", True)
                prompt = (
                    f"Task: {task.get('task_description', '')}\n"
                    f"Available apps: {', '.join(available)}\n"
                    f"Observation: {observation}\n\n"
                    "What is the next action?"
                )
                out.append(
                    DataSample(
                        question=prompt,
                        target=expected,
                        context=observation,
                        others={
                            "task_id": task_id,
                            "step_index": i,
                            "is_correct": is_correct,
                        },
                    )
                )
        return out

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Loose match: case-fold, strip, and check substring containment."""
        p = (predicted or "").strip().lower()
        g = (ground_truth or "").strip().lower()
        if not p or not g:
            return False
        return p == g or p in g or g in p

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


def require_appworld() -> bool:
    """Return True if appworld is importable; otherwise print a hint.

    ``bench_appworld`` short-circuits when this returns False so the
    bench runner doesn't import appworld at module-load time.
    """
    try:
        import appworld  # noqa: F401

        return True
    except ImportError:
        print(
            "appworld is not installed. To run a real AppWorld eval:\n"
            "  pip install appworld\n"
            "  export ALFW_API_KEY=...\n"
            "  ceng bench appworld --limit 30"
        )
        return False


def seed_playbook() -> str:
    """Curated multi-turn-agent seed playbook for AppWorld.

    ~25 strategies across the seven ACE sections.
    """
    return _APPWORLD_SEED_PLAYBOOK


_APPWORLD_SEED_PLAYBOOK = """## Strategies & Insights
[str-00001] helpful=0 harmful=0 :: AppWorld tasks usually require reading files, sending emails, or updating a database. Plan the action BEFORE executing.
[str-00002] helpful=0 harmful=0 :: AppWorld tasks are multi-step. Identify ALL the sub-tasks from the task description, complete them in a logical order, then verify.
[str-00003] helpful=0 harmful=0 :: When a task description is ambiguous, prefer the simpler interpretation. AppWorld tasks are graded strictly.
[str-00004] helpful=0 harmful=0 :: Use the help() function on any unfamiliar app — it returns the canonical usage pattern.

## Formulas & Calculations
[cal-00001] helpful=0 harmful=0 :: When a task says 'the most recent', filter by timestamp desc and take the top result.
[cal-00002] helpful=0 harmful=0 :: When a task says 'all matching X', don't paginate — just collect them all and return the union.

## Code Snippets & Templates
[ctx-00001] helpful=0 harmful=0 :: Action format: <app_name>.<method>(<arg>=<value>). Read the app's API spec carefully.
[ctx-00002] helpful=0 harmful=0 :: When chaining actions, prefer composing them in a single call when the API supports it (e.g. update with multiple fields) instead of multiple round-trips.

## Common Mistakes To Avoid
[mis-00001] helpful=0 harmful=0 :: Don't guess the app's API; the names matter. 'phone' vs 'telephone' vs 'contacts' are different apps with different APIs.
[mis-00002] helpful=0 harmful=0 :: Don't assume file paths; check the working directory first.
[mis-00003] helpful=0 harmful=0 :: Don't ignore errors from previous actions; they cascade.

## Problem-Solving Heuristics
[ps-00001] helpful=0 harmful=0 :: When stuck, list the directory or run a search query before guessing.
[ps-00002] helpful=0 harmful=0 :: When the task involves multiple recipients, build the list explicitly rather than assuming a default.

## Context Clues & Indicators
[cc-00001] helpful=0 harmful=0 :: AppWorld tasks often contain a 'task description' that names the entities involved — extract them first.
[cc-00002] helpful=0 harmful=0 :: 'Test' apps (supervisor, task) are usually metadata-only; real work goes in the domain apps (email, file_system, etc.).

## Others
[oth-00001] helpful=0 harmful=0 :: When the task is complete, call the supervisor's mark_task_complete() function. Otherwise the score is 0 even if the side effects are right.
"""
