"""Compress a long message with PPA using a real model (see common.py)."""

import common

from ceng import Context, Message, Role


def main() -> None:
    with common.model_runtime() as runtime:
        context = Context(
            (
                Message(Role.SYSTEM, "You are a careful analyst."),
                Message(Role.USER, common.long_document()),
            ),
            runtime,
        )
        smaller = context.compress("ppa", budget=400)
        report = smaller.report
        print(f"{report.original_tokens} -> {report.final_tokens} tokens")
        print(
            f"steps: {len(report.steps)}, LLM tokens: {report.usage.total_tokens}"
        )
        print(smaller.messages[-1].content[:300])


if __name__ == "__main__":
    main()
