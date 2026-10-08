"""Grounded answers: citations that are verified against the source."""

from foveate.grounding.answer import NOT_FOUND, Answer, Citation
from foveate.grounding.quotes import normalise, supports

__all__ = ["NOT_FOUND", "Answer", "Citation", "normalise", "supports"]
