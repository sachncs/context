"""Samples and graders shared by evolution and benchmarks."""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    """One task instance.

    Attributes:
        question: What the model is asked.
        target: Gold answer used for grading.
        context: Supporting material shown with the question.
        id: Stable identifier (may be empty).
    """

    question: str
    target: str
    context: str = ""
    id: str = ""


class Grader(abc.ABC):
    """Decides whether a model answer matches the gold answer."""

    @abc.abstractmethod
    def is_correct(self, predicted: str, target: str) -> bool:
        """Returns whether `predicted` is acceptable for `target`."""

    def feedback(self, predicted: str, target: str) -> str:
        """Returns environment feedback describing a wrong answer."""
        return f"model answer was {predicted!r}; expected {target!r}"
