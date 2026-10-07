"""Compress a long message with PPA, offline (scripted model)."""

from ceng import Context, Message, Role, Runtime
from ceng.backends import ScriptedBackend
from ceng.cache import MemoryCache


def main() -> None:
    # Swap the backend for Runtime.from_env() to use a real provider.
    runtime = Runtime(
        backend=ScriptedBackend(
            default=lambda request: "Key facts: " + "x" * 80
        ),
        cache=MemoryCache(),
    )
    document = " ".join(f"Fact {i}: the value is {i * 7}." for i in range(400))
    context = Context(
        (
            Message(Role.SYSTEM, "You are a careful analyst."),
            Message(Role.USER, document),
        ),
        runtime,
    )

    smaller = context.compress("ppa", budget=400, leaf_tokens=256)

    report = smaller.report
    print(f"{report.original_tokens} -> {report.final_tokens} tokens")
    print(
        f"steps: {len(report.steps)}, LLM tokens: {report.usage.total_tokens}"
    )
    assert smaller.token_count <= 400


if __name__ == "__main__":
    main()
