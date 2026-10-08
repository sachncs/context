"""Parsing JSON out of model output."""

from __future__ import annotations

import json
import re

from foveate import errors

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict[str, object]:
    """Parses a JSON object out of model text, tolerating code fences.

    Raises:
        ValidationError: If no JSON object can be recovered.
    """
    cleaned = FENCE.sub("", text.strip())
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise errors.ValidationError(
        f"no JSON object in model output: {text[:80]!r}"
    )
