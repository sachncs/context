"""Shrink a long conversation and a bulky tool result with no model.

Every method here is offline: `window`, `extractive`, `selective`, `query`,
`tool_output` and `clear_tool_results`.
"""

import json

from foveate import Context, Message, Role, Runtime

runtime = Runtime.without_llm()  # any model call would raise


def chat() -> Context:
    messages = []
    for turn in range(30):
        messages.append(Message(Role.USER, f"turn {turn}: " + "filler " * 80))
        messages.append(Message(Role.ASSISTANT, "noted. " * 40))
    messages[10] = Message(Role.USER, "Our deployment region is eu-west-1.")
    messages.append(Message(Role.USER, "Which deployment region did I say?"))
    return Context(tuple(messages), runtime)


def tool_context() -> Context:
    rows = [{"id": i, "status": "ok", "note": "x" * 40} for i in range(400)]
    return Context(
        (
            Message(Role.USER, "List the orders."),
            Message(Role.TOOL, json.dumps({"rows": rows}), name="orders"),
        ),
        runtime,
    )


def main() -> None:
    conversation = chat()
    print(f"conversation: {conversation.token_count:,} tokens")
    for method in ("window", "extractive", "selective", "query"):
        options = {"query": "deployment region"} if method == "query" else {}
        out = conversation.compress(method, budget=600, **options)
        kept = any("eu-west-1" in m.content for m in out.messages)
        print(
            f"  {method:10s} -> {out.token_count:4d} tokens, fact kept: {kept}"
        )

    tool = tool_context()
    reduced = tool.compress("tool_output", budget=300)
    print(f"tool result: {tool.token_count:,} -> {reduced.token_count} tokens")
    print(" ", reduced.messages[-1].content[:90], "...")


if __name__ == "__main__":
    main()
