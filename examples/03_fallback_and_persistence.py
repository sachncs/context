"""Fallback when the model is unreachable; save and load as an OKF bundle."""

import tempfile
from pathlib import Path

import common

from ceng import Context, Message, Role, Runtime
from ceng.backends import OpenAIBackend, ResilientBackend, RetryPolicy


def main() -> None:
    # A real backend pointed at a port nobody listens on: every call fails.
    dead = Runtime(
        backend=ResilientBackend(
            OpenAIBackend(base_url="http://127.0.0.1:9/v1", api_key="unused"),
            retry=RetryPolicy(attempts=1),
            timeout=2.0,
        ),
        model="any-model",
    )
    context = Context((Message(Role.USER, common.long_document()),), dead)

    safe = context.compress("ppa|extractive", budget=300)
    print(
        "degraded to:",
        [s.name for s in safe.report.steps if "fallback" in s.name],
    )

    with tempfile.TemporaryDirectory() as scratch:
        safe.save(Path(scratch) / "bundle")  # OKF markdown bundle
        again = Context.load(Path(scratch) / "bundle", runtime=dead)
        assert again.messages == safe.messages and again.report == safe.report
        print(
            "round-trip ok:",
            sorted(p.name for p in (Path(scratch) / "bundle").rglob("*.md")),
        )


if __name__ == "__main__":
    main()
