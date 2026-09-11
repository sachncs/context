"""Example 01 — quickest path to a compressed message.

Run with:  python examples/01_quickstart.py

Requires only ``litellm`` (already a ceng dependency). Configure an
``OPENAI_API_KEY`` (or any litellm-routable credential) before
running. The script falls back to a stub backend when no key is
present so it stays runnable in CI / local sandbox without network.
"""

from __future__ import annotations

import os
import sys

import ceng
from ceng import ppa_compress


SAMPLE_DOCUMENT = (
    " ".join(
        f"Section {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        f"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        f"nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
        f"reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        f"pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
        f"culpa qui officia deserunt mollit anim id est laborum."
        for i in range(1, 21)
    )
)


class StubBackend:
    """Trivial backend so the example runs without any LLM credential."""

    name = "stub"

    def complete(self, messages, model, **kw):
        return (
            "User is choosing between Kafka and RabbitMQ for a 10k "
            "events/sec event bus, prefers managed services, and "
            "needs to lock the decision by end of next month."
        )


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

    messages = [
        {"role": "system", "content": "You are a careful analyst."},
        {"role": "user", "content": SAMPLE_DOCUMENT},
    ]
    small = ppa_compress(
        messages,
        budget_tokens=200,
        llm="gpt-4o-mini",
        cache_dir="",
        backend=backend,
    )
    print(small[1]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
