"""Stable fingerprints for cache keys and identifiers."""

from __future__ import annotations

import hashlib
import json


def fingerprint(*parts: object) -> str:
    """Returns a stable SHA-256 hex digest of JSON-serialisable parts.

    Args:
        *parts: Values that `json.dumps` can encode. Key order inside
            mappings does not affect the result.

    Returns:
        A 64-character hex digest.
    """
    encoded = json.dumps(
        parts, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
