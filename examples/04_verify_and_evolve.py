"""Macro-fallacy check and ACE playbook evolution, offline."""

import json

from ceng import Context, Message, Role, Runtime
from ceng.backends import ScriptedBackend
from ceng.cache import MemoryCache
from ceng.evolution import Evolver, Grader, Playbook, Sample


class ExactMatch(Grader):
    def is_correct(self, predicted: str, target: str) -> bool:
        return predicted == target


def respond(request) -> str:
    text = request.messages[-1].content
    if "analysis expert" in text:  # Generator
        return json.dumps(
            {"reasoning": "...", "bullet_ids": [], "final_answer": "5"}
        )
    if "diagnose why" in text:  # Reflector
        return json.dumps({"key_insight": "Add the numbers before answering."})
    if "master curator" in text:  # Curator
        return json.dumps(
            {
                "operations": [
                    {
                        "type": "ADD",
                        "section": "strategies_and_insights",
                        "content": "Add the numbers before answering.",
                    }
                ]
            }
        )
    return "0.5"  # probability questions


def main() -> None:
    runtime = Runtime(
        backend=ScriptedBackend(default=respond), cache=MemoryCache()
    )
    context = Context((Message(Role.USER, "hi"),), runtime)

    verdict = context.verify(
        "macro_fallacy",
        question="Do users like it?",
        population="all users",
        tree=[
            {"description": "new", "prior": 0.4},
            {"description": "returning", "prior": 0.6},
        ],
    )
    print("consistent:", verdict.passed, verdict.detail)

    result = Evolver(runtime).evolve(
        Playbook(), [Sample("2+2?", "4")], ExactMatch()
    )
    print("accuracy:", result.accuracy, "bullets:", len(result.playbook))
    print(result.playbook.render().strip().splitlines()[1])


if __name__ == "__main__":
    main()
