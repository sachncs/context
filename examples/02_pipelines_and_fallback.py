"""Compose strategies: pipelines with '+', degraded-mode fallback with '|'."""

from ceng import Context, Message, Role, Runtime, errors
from ceng.backends import ScriptedBackend
from ceng.cache import MemoryCache


def conversation(runtime: Runtime) -> Context:
    turns = [
        Message(
            Role.USER if i % 2 == 0 else Role.ASSISTANT,
            f"turn {i}: " + "word " * 120,
        )
        for i in range(20)
    ]
    return Context(tuple(turns), runtime)


def main() -> None:
    healthy = Runtime(
        backend=ScriptedBackend(default=lambda request: "summary " * 10),
        cache=MemoryCache(),
    )
    # Keep the first 2 and last 3 turns; summarise the rest.
    compact = conversation(healthy).compress(
        "ushape", budget=900, head=2, tail=3
    )
    print(
        "ushape:",
        compact.report.original_tokens,
        "->",
        compact.report.final_tokens,
    )

    # Cheap stage first, precise stage only if still over budget.
    chained = conversation(healthy).compress("window+ushape", budget=900)
    print("pipeline label:", chained.report.method)

    # The model is down: fall back to LLM-free extractive compression.
    down = Runtime(
        backend=ScriptedBackend(
            [errors.PermanentBackendError("provider outage")]
        ),
        cache=MemoryCache(),
    )
    safe = conversation(down).compress("ppa|extractive", budget=900)
    print(
        "fallback steps:",
        [step.name for step in safe.report.steps if "fallback" in step.name],
    )


if __name__ == "__main__":
    main()
