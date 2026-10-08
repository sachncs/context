"""Context verification."""

from __future__ import annotations

from foveate import errors
from foveate.verification.base import Verdict, Verifier
from foveate.verification.fits import FitsBudget
from foveate.verification.macro import (
    LeafEstimate,
    MacroFallacy,
    MacroVerdict,
    TreeNode,
    parse_probability,
)


def resolve(method: str | Verifier, **options: object) -> Verifier:
    """Turns a method specification into a `Verifier`.

    For `macro_fallacy`, `tree` may be given as nested mappings and is
    converted with `TreeNode.build`.

    Raises:
        ConfigError: For unknown methods or invalid options.
    """
    if isinstance(method, Verifier):
        if options:
            raise errors.ConfigError(
                "options cannot be combined with a Verifier instance"
            )
        return method
    cls = Verifier.registry.get(method)
    tree = options.get("tree")
    if cls is MacroFallacy and isinstance(tree, list):
        options = {**options, "tree": TreeNode.build(tree)}
    try:
        return cls(**options)
    except TypeError as exc:
        raise errors.ConfigError(
            f"invalid options for {method!r}: {exc}"
        ) from exc


__all__ = [
    "FitsBudget",
    "LeafEstimate",
    "MacroFallacy",
    "MacroVerdict",
    "TreeNode",
    "Verdict",
    "Verifier",
    "parse_probability",
    "resolve",
]
