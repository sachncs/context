"""Example 02 — compress a long message into an OKF bundle.

Run with:  python examples/02_okf_bundle.py

Requires the ``openai`` extra (or any litellm-routable credential)
and a writable current directory. The script writes a bundle under
``./ctx-out/my-context/`` containing ``index.md`` (always) and
``combined.md`` (when the input is compressible). Pass
``index_only=False`` to also materialise per-leaf files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ceng
from ceng import ppa_compress_to_okf


SAMPLE_DOCUMENT = (
    "Chapter 1: We open with the protagonist's morning routine in a "
    "small seaside town. She buys fish from the same vendor her "
    "mother did. The chapter closes with a phone call from her "
    "sister that she does not answer.\n\n"
    "Chapter 2: Flashback to the summer she turned fifteen. A "
    "house fire, a stolen bicycle, a first love that does not last "
    "the season. The town watches but does not interfere.\n\n"
    "Chapter 3: Present day. She runs a print shop that nobody "
    "visits. The phone keeps ringing. The vendor tells her, in the "
    "way small-town vendors do, that her mother would have wanted "
    "her to answer.\n\n"
    "Chapter 4: A letter arrives from an address she does not "
    "recognise. The handwriting is her father's, who left when she "
    "was three. She reads it twice, folds it into her apron pocket, "
    "and walks down to the pier.\n\n"
    "Chapter 5: Closing scene. She lights a cigarette she has been "
    "saving for twenty years. The phone rings one more time. She "
    "lets it ring."
)


class StubBackend:
    name = "stub"

    def complete(self, messages, model, **kw):
        return (
            "A woman in a seaside town confronts an unsent letter "
            "from her estranged father while ignoring her sister's "
            "phone calls."
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

    bundle_dir = Path("ctx-out")
    bundle_dir.mkdir(exist_ok=True)
    concepts = ppa_compress_to_okf(
        [
            {"role": "user", "content": SAMPLE_DOCUMENT},
        ],
        bundle_dir=str(bundle_dir),
        bundle_name="my-context",
        budget_tokens=200,
        llm="gpt-4o-mini",
        cache_dir="",
        backend=backend,
    )
    print(f"wrote {len(concepts)} concepts to {bundle_dir / 'my-context'}")
    print("contents:")
    for c in concepts:
        print(f"  - {c.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
