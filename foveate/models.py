"""Known model context windows, used for `budget="auto"`.

Values are the publicly documented maximum context sizes in tokens. Hosting
providers sometimes serve a smaller window than the model supports, so an
explicit `Runtime(context_window=...)` or `FOVEATE_CONTEXT_WINDOW` always
wins over this table.
"""

from __future__ import annotations

import dataclasses

from foveate import errors


@dataclasses.dataclass(frozen=True, slots=True)
class ModelInfo:
    """What foveate knows about a model.

    Attributes:
        prefix: Model-name prefix this entry matches (longest wins).
        context_window: Maximum input+output tokens.
    """

    prefix: str
    context_window: int


KNOWN = (
    ModelInfo("gpt-4o", 128_000),
    ModelInfo("gpt-4-turbo", 128_000),
    ModelInfo("gpt-4.1", 1_047_576),
    ModelInfo("gpt-3.5-turbo", 16_385),
    ModelInfo("o1", 200_000),
    ModelInfo("o3", 200_000),
    ModelInfo("o4-mini", 200_000),
    ModelInfo("openai/gpt-oss", 131_072),
    ModelInfo("gpt-oss", 131_072),
    ModelInfo("claude-3", 200_000),
    ModelInfo("claude-sonnet-4", 200_000),
    ModelInfo("claude-opus-4", 200_000),
    ModelInfo("gemini-1.5-pro", 2_097_152),
    ModelInfo("gemini-1.5-flash", 1_048_576),
    ModelInfo("gemini-2", 1_048_576),
    ModelInfo("llama-3.1", 131_072),
    ModelInfo("llama-3.3", 131_072),
    ModelInfo("meta/llama-3.1", 131_072),
    ModelInfo("mistral-large", 131_072),
    ModelInfo("qwen2.5", 32_768),
    ModelInfo("deepseek-v3", 131_072),
    ModelInfo("deepseek-ai/deepseek-v", 131_072),
)
MIN_RESERVE = 1_024


def lookup(model: str) -> ModelInfo | None:
    """Returns the best (longest-prefix) match for `model`, or None."""
    lowered = model.lower()
    candidates = [
        info
        for info in KNOWN
        if lowered.split("/")[-1].startswith(info.prefix)
        or lowered.startswith(info.prefix)
    ]
    return max(candidates, key=lambda info: len(info.prefix), default=None)


def auto_budget(
    model: str,
    context_window: int | None = None,
    output_reserve: int = 2_000,
) -> int:
    """Returns how many input tokens a prompt may use.

    The budget is the window minus room for the answer and a 5% safety
    margin for tokenizer differences (counts are estimates).

    Args:
        model: Model id.
        context_window: Explicit window; otherwise looked up by name.
        output_reserve: Tokens kept free for the answer.

    Raises:
        ConfigError: If the window is unknown and was not given.
    """
    window = context_window
    if window is None:
        info = lookup(model)
        if info is None:
            raise errors.ConfigError(
                f"unknown context window for {model!r}; pass budget=... or "
                "Runtime(context_window=...)"
            )
        window = info.context_window
    budget = int(window * 0.95) - max(output_reserve, MIN_RESERVE)
    if budget < 1:
        raise errors.ConfigError(f"context window {window} is too small")
    return budget
