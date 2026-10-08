"""LLM-free compression of tool results (JSON, logs, CSV, HTML).

Agents often receive huge structured tool outputs. A `Reducer` shrinks one
format while keeping its shape (keys, header, counts), so the model can still
see what was cut. `ToolOutput` tries ever tighter levels until the context
fits the budget.
"""

from __future__ import annotations

import abc
import csv
import dataclasses
import html
import io
import json
import re
from typing import TYPE_CHECKING, ClassVar

from foveate import errors
from foveate import messages as messages_lib
from foveate.compression import base, report
from foveate.internals import registry

if TYPE_CHECKING:
    from foveate import context as context_lib

LEVELS = 6
BASE_ITEMS = 32
BASE_STRING = 800
BASE_LINES = 64
TAG = re.compile(r"<[^>]+>")
BLOCK = re.compile(
    r"<(script|style|head)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
SPACES = re.compile(r"[ \t]+")
DIGITS = re.compile(r"\d+")


def scale(base_value: int, level: int) -> int:
    """Halves `base_value` per level, never below 1."""
    return max(1, base_value >> level)


class Reducer(abc.ABC):
    """Shrinks one text format; higher `level` means smaller output."""

    registry: ClassVar[registry.Registry[type[Reducer]]] = registry.Registry(
        "tool-output reducer"
    )

    @abc.abstractmethod
    def matches(self, text: str) -> bool:
        """Returns whether `text` is in this reducer's format."""

    @abc.abstractmethod
    def reduce(self, text: str, level: int) -> str:
        """Returns a smaller rendering of `text` at the given level."""


def prune(value: object, level: int) -> object:
    """Recursively truncates arrays and strings, noting what was cut."""
    items = scale(BASE_ITEMS, level)
    chars = scale(BASE_STRING, level)
    if isinstance(value, dict):
        return {key: prune(item, level) for key, item in value.items()}
    if isinstance(value, list):
        kept = [prune(item, level) for item in value[:items]]
        if len(value) > items:
            kept.append(f"... {len(value) - items} more items")
        return kept
    if isinstance(value, str) and len(value) > chars:
        return f"{value[:chars]}... [{len(value) - chars} more chars]"
    return value


@Reducer.registry.register("json")
class JsonReducer(Reducer):
    """Truncates long arrays and strings; keeps every key."""

    def matches(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return False
        try:
            json.loads(stripped)
        except ValueError:
            return False
        return True

    def reduce(self, text: str, level: int) -> str:
        pruned = prune(json.loads(text), level)
        return json.dumps(pruned, ensure_ascii=False, separators=(",", ":"))


@Reducer.registry.register("csv")
class CsvReducer(Reducer):
    """Keeps the header and the first and last rows."""

    def matches(self, text: str) -> bool:
        lines = text.strip().splitlines()
        if len(lines) < 3 or text.lstrip().startswith(("{", "[", "<")):
            return False
        width = lines[0].count(",")
        return width > 0 and all(
            line.count(",") == width for line in lines[1:4]
        )

    def reduce(self, text: str, level: int) -> str:
        rows = list(csv.reader(io.StringIO(text.strip())))
        header, body = rows[0], rows[1:]
        keep = scale(BASE_LINES, level)
        head, tail = body[: keep // 2 + keep % 2], body[-(keep // 2) :]
        if keep // 2 == 0:
            tail = []
        omitted = len(body) - len(head) - len(tail)
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(head)
        if omitted > 0:
            out.write(f"... {omitted} rows omitted\n")
        writer.writerows(tail)
        return out.getvalue().rstrip("\n")


@Reducer.registry.register("html")
class HtmlReducer(Reducer):
    """Strips markup, scripts and styles, leaving the visible text."""

    def matches(self, text: str) -> bool:
        head = text.lstrip()[:200].lower()
        return head.startswith(("<!doctype", "<html", "<body", "<div"))

    def reduce(self, text: str, level: int) -> str:
        visible = html.unescape(TAG.sub(" ", BLOCK.sub(" ", text)))
        lines = [SPACES.sub(" ", line).strip() for line in visible.splitlines()]
        out = "\n".join(line for line in lines if line)
        limit = scale(BASE_STRING * 4, level)
        if len(out) > limit:
            out = f"{out[:limit]}... [{len(out) - limit} more chars]"
        return out


@Reducer.registry.register("log")
class LogReducer(Reducer):
    """Collapses repeated lines (ignoring digits) and keeps head and tail."""

    def matches(self, text: str) -> bool:
        return text.count("\n") >= 8

    def reduce(self, text: str, level: int) -> str:
        collapsed: list[str] = []
        previous, repeats = "", 0
        for line in text.splitlines():
            key = DIGITS.sub("#", line)
            if collapsed and key == previous:
                repeats += 1
                continue
            if repeats:
                collapsed.append(f"    (repeated {repeats} more times)")
            collapsed.append(line)
            previous, repeats = key, 0
        if repeats:
            collapsed.append(f"    (repeated {repeats} more times)")
        keep = scale(BASE_LINES, level)
        if len(collapsed) > keep:
            half = max(1, keep // 2)
            omitted = len(collapsed) - 2 * half
            collapsed = [
                *collapsed[:half],
                f"... {omitted} lines omitted",
                *collapsed[-half:],
            ]
        return "\n".join(collapsed)


def reduce_text(text: str, level: int) -> str:
    """Applies the first matching reducer; unknown formats pass through."""
    for name in Reducer.registry.names():
        reducer = Reducer.registry.get(name)()
        if reducer.matches(text):
            try:
                return reducer.reduce(text, level)
            except (ValueError, csv.Error):
                continue
    return text


@base.Compressor.register("tool_output")
@dataclasses.dataclass(frozen=True)
class ToolOutput(base.Compressor):
    """Shrinks structured tool results in place, keeping their shape.

    Messages are reduced at progressively tighter levels until the context
    fits; system messages are never touched. No LLM calls are made.

    Attributes:
        roles: Roles whose messages are reduced.
    """

    roles: tuple[messages_lib.Role, ...] = (
        messages_lib.Role.TOOL,
        messages_lib.Role.ASSISTANT,
        messages_lib.Role.USER,
    )

    def __post_init__(self) -> None:
        if not self.roles:
            raise errors.ConfigError("roles must not be empty")

    async def run(
        self,
        context: context_lib.Context,
        budget: report.Budget,
        trace: report.Trace,
    ) -> context_lib.Context:
        result = context
        for level in range(LEVELS):
            messages = tuple(
                m.with_content(reduce_text(m.content, level))
                if m.role in self.roles
                else m
                for m in context.messages
            )
            result = dataclasses.replace(context, messages=messages)
            if result.token_count <= budget.tokens:
                break
        return result
