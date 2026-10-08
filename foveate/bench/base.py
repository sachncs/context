"""Benchmark interface: data loading, prompting and grading in one object."""

from __future__ import annotations

import abc
import importlib.resources
import json
import pathlib
from collections.abc import Callable, Mapping
from typing import ClassVar

from foveate import errors
from foveate import messages as messages_lib
from foveate.evolution import grading
from foveate.evolution import playbook as playbook_lib
from foveate.internals import registry

ANSWER_PREFIXES = ("Answer:", "Output:", "Result:")
INSTRUCTION = "Answer with ONLY the final answer value, no explanation."


class Benchmark(grading.Grader):
    """A task family: how to load samples, prompt a model and grade it.

    Subclasses define `name`, `fixture` and `row_to_sample`, and may
    override grading and answer extraction.
    """

    registry: ClassVar[registry.Registry[type[Benchmark]]] = registry.Registry(
        "benchmark"
    )
    name: ClassVar[str] = ""
    cited_baseline: ClassVar[float | None] = None
    cited_ace: ClassVar[float | None] = None

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[Benchmark]], type[Benchmark]]:
        """Returns a class decorator registering a benchmark under `name`."""

        def decorator(subclass: type[Benchmark]) -> type[Benchmark]:
            cls.registry.add(name, subclass)
            subclass.name = name
            return subclass

        return decorator

    @abc.abstractmethod
    def row_to_sample(self, row: Mapping[str, object]) -> grading.Sample:
        """Converts one dataset row into a `Sample`.

        Raises:
            ValidationError: If the row lacks required fields.
        """

    def read_rows(
        self, path: pathlib.Path | None
    ) -> list[Mapping[str, object]]:
        """Reads JSONL rows from `path`, or from the packaged fixture."""
        try:
            if path is None:
                resource = (
                    importlib.resources.files("foveate.bench")
                    .joinpath("fixtures")
                    .joinpath(f"{self.name}.jsonl")
                )
                text = resource.read_text(encoding="utf-8")
            else:
                text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise errors.ValidationError(
                f"cannot read {self.name} data: {exc}"
            ) from exc
        rows: list[Mapping[str, object]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise errors.ValidationError(
                    f"{self.name} line {number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise errors.ValidationError(
                    f"{self.name} line {number}: expected an object"
                )
            rows.append(row)
        return rows

    def load_samples(
        self, limit: int | None = None, path: pathlib.Path | None = None
    ) -> list[grading.Sample]:
        """Loads up to `limit` samples from `path` or the packaged fixture.

        Raises:
            ValidationError: For unreadable or malformed data.
            ConfigError: If `limit` is not positive.
        """
        if limit is not None and limit < 1:
            raise errors.ConfigError("limit must be >= 1")
        rows = self.read_rows(path)
        chosen = rows if limit is None else rows[:limit]
        samples = []
        for index, row in enumerate(chosen):
            try:
                samples.append(self.row_to_sample(row))
            except errors.ValidationError as exc:
                raise errors.ValidationError(
                    f"{self.name} row {index}: {exc}"
                ) from exc
        return samples

    def seed_playbook(self) -> playbook_lib.Playbook:
        """Returns the curated starting playbook shipped for this task."""
        resource = (
            importlib.resources.files("foveate.bench")
            .joinpath("seeds")
            .joinpath(f"{self.name}.md")
        )
        return playbook_lib.Playbook.parse(resource.read_text(encoding="utf-8"))

    def build_messages(
        self,
        sample: grading.Sample,
        playbook: playbook_lib.Playbook | None,
    ) -> tuple[messages_lib.Message, ...]:
        """Builds the prompt; a playbook, when given, is injected."""
        prompt = (
            f"Question: {sample.question}\n"
            f"Context: {sample.context}\n{INSTRUCTION}"
        )
        messages = [messages_lib.Message(messages_lib.Role.USER, prompt)]
        if playbook is not None and len(playbook):
            messages.insert(0, playbook.as_message())
        return tuple(messages)

    def extract_answer(self, text: str) -> str:
        """Reduces a model reply to its answer (first non-empty line)."""
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            for prefix in ANSWER_PREFIXES:
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
            return line
        return text.strip()


def required_text(row: Mapping[str, object], *names: str) -> str:
    """Returns the first present field among `names` as a string.

    Unlike truthiness-based lookups, a legitimate falsy value such as the
    integer `0` is kept.

    Raises:
        ValidationError: If none of the fields is present.
    """
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return str(value).strip()
    raise errors.ValidationError(f"missing field (one of {', '.join(names)})")
