"""Example 03 — long-chat compaction with U-shape preservation.

Run with:  python examples/03_compact_chat.py

Demonstrates ``compact_messages`` on a synthetic 20-message chat
that crosses the budget; the middle strip is summarised in a single
backend call while the head and tail are kept verbatim.
"""

from __future__ import annotations

import os
import sys

import ceng
from ceng.compact import compact_messages


SYSTEM = {"role": "system", "content": "You are an SRE debugging a payment outage."}

LONG_CHAT = (
    [SYSTEM]
    + [
        {"role": "user", "content": "The latency on /charge spiked to 5s at 14:02 UTC."},
        {"role": "assistant", "content": "I'll check the gateway logs."},
    ]
    + [
        {"role": "user", "content": f"Investigator note #{i}: the gateway returned 200 but the body was empty {i} times in a row."}
        for i in range(12)
    ]
    + [
        {"role": "assistant", "content": "Found it — the proxy buffer was set to 8k and we crossed it with the new payload schema."},
        {"role": "user", "content": "Rolling back to the previous buffer size now."},
        {"role": "assistant", "content": "Latency back to 80ms p99. Closing the incident."},
    ]
)


class StubBackend:
    name = "stub"

    def complete(self, messages, model, **kw):
        return (
            "Investigators traced the latency spike to an 8k proxy "
            "buffer being exceeded by a payload schema change; "
            "rolling back the buffer size restored p99 to 80ms."
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

    print(f"before: {len(LONG_CHAT)} messages")
    compacted, prov = compact_messages(
        LONG_CHAT,
        backend=backend,
        llm="gpt-4o-mini",
        cache_dir="",
        preserve_first=2,
        preserve_last=4,
        summarise_middle=True,
    )
    print(f"after:  {len(compacted)} messages")
    print(f"  preserved head: {prov.preserved_first}")
    print(f"  preserved tail: {prov.preserved_last}")
    print(f"  summarised middle: {prov.summarised_count}")
    print()
    print("compacted messages:")
    for m in compacted:
        print(f"  [{m['role']}] {m['content'][:80]}{'...' if len(m['content']) > 80 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
