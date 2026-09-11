"""Example 04 — ACE Evolver smoke run.

Run with:  python examples/04_evolver_smoke.py

Runs the Generator -> Reflector -> Curator loop over three synthetic
samples and prints the bullets that the Curator added. Requires
``litellm``; falls back to a stub backend when no credential is set
so the script still runs end-to-end.
"""

from __future__ import annotations

import json
import os
import sys

import ceng
from ceng.playbook import empty_playbook
from ceng.playbook.evolver import Evolver


SAMPLES = [
    {
        "question": "What is 12 * 7?",
        "ground_truth": "84",
        "context": "Direct arithmetic question.",
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "context": "Geography question.",
    },
    {
        "question": "Spell 'receivable' backwards.",
        "ground_truth": "elbicaecer",
        "context": "Spelling question.",
    },
]


class StubBackend:
    """A stub backend that emits valid Generator / Reflector / Curator
    JSON so the Evolver loop completes without a real LLM call."""

    name = "stub"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, model, **kw):
        self.call_count += 1
        system = messages[0]["content"].lower() if messages else ""
        if "generator" in system:
            return json.dumps(
                {
                    "reasoning": "Use the playbook entry if any.",
                    "bullet_ids": [],
                    "final_answer": "84",
                }
            )
        if "reflector" in system:
            return json.dumps(
                {
                    "reasoning": "The model produced a plausible answer.",
                    "error_identification": "",
                    "root_cause_analysis": "",
                    "correct_approach": "Recite the formula.",
                    "key_insight": "Direct arithmetic",
                    "bullet_tags": [],
                }
            )
        if "curator" in system:
            return json.dumps(
                {
                    "reasoning": "Add a strategy bullet.",
                    "operations": [
                        {
                            "type": "ADD",
                            "section": "strategies_and_insights",
                            "content": "Prefer direct computation over lookup tables.",
                        }
                    ],
                }
            )
        return "{}"


def evaluator(question: str, answer: str, sample: dict) -> str:
    return "correct" if answer == sample["ground_truth"] else f"wrong: {answer!r}"


def main() -> int:
    if os.environ.get("OPENAI_API_KEY"):
        ceng.set_backend("litellm")
        backend = None
    else:
        print(
            "warning: OPENAI_API_KEY not set; using StubBackend so the "
            "script still runs end-to-end without a credential",
            file=sys.stderr,
        )
        backend = StubBackend()

    ev = Evolver(backend=backend, llm="gpt-4o-mini", cache_dir="")
    playbook, stats = ev.run(
        playbook=empty_playbook(),
        queries=SAMPLES,
        evaluator=evaluator,
        max_iterations=1,
    )
    print(f"backend calls: {getattr(backend, 'call_count', 'n/a')}")
    print(f"steps recorded: {len(stats)}")
    print(f"bullets after run: {len(playbook.bullets)}")
    for b in playbook.bullets.values():
        print(f"  [{b.id}] {b.section}: {b.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
