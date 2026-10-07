"""Compress without any model: window, truncate, extractive, offload."""

import common

from ceng import Context, Message, Role, Runtime
from ceng.stores import MemoryNotesStore


def main() -> None:
    runtime = Runtime.without_llm()  # any model call would raise
    document = Context((Message(Role.USER, common.long_document()),), runtime)
    for method in ("truncate", "extractive"):
        out = document.compress(method, budget=200)
        print(f"{method:10s} {out.report.original_tokens} -> {out.token_count}")

    chat = Context(
        tuple(
            Message(
                Role.USER if i % 2 == 0 else Role.ASSISTANT,
                f"turn {i} " + "word " * 120,
            )
            for i in range(12)
        ),
        runtime,
    )
    print("window     ", chat.compress("window", budget=400).token_count)
    print(
        "window+extractive",
        chat.compress("window+extractive", budget=400).token_count,
    )

    store = MemoryNotesStore()  # nothing is lost: old turns move to the store
    slim = chat.compress("offload", budget=400, store=store, head=1, tail=2)
    print(slim.messages[1].content)
    print("recoverable chars:", len(store.entries()[0].content))


if __name__ == "__main__":
    main()
