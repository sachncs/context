"""Tokenizer interface and built-in implementations."""

from __future__ import annotations

import abc
import math

from ceng import errors
from ceng.internals import registry


class Tokenizer(abc.ABC):
    """Counts tokens and splits text on token boundaries."""

    registry: registry.Registry[type[Tokenizer]] = registry.Registry(
        "tokenizer"
    )

    @abc.abstractmethod
    def count(self, text: str) -> int:
        """Returns the number of tokens in `text`."""

    @abc.abstractmethod
    def truncate(self, text: str, max_tokens: int) -> str:
        """Returns the longest prefix of `text` within `max_tokens`."""


@Tokenizer.registry.register("heuristic")
class HeuristicTokenizer(Tokenizer):
    """Dependency-free estimator: roughly `chars_per_token` characters each.

    Attributes:
        chars_per_token: Average characters per token (default 4.0, which
            matches English prose on common BPE vocabularies).
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """Creates the estimator.

        Raises:
            ConfigError: If `chars_per_token` is not positive.
        """
        if chars_per_token <= 0:
            raise errors.ConfigError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return text[: int(max_tokens * self.chars_per_token)]


@Tokenizer.registry.register("tiktoken")
class TiktokenTokenizer(Tokenizer):
    """Exact BPE counts through `tiktoken` (optional dependency).

    Attributes:
        encoding_name: tiktoken encoding, e.g. "cl100k_base".
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Loads the encoding.

        Raises:
            ConfigError: If tiktoken is missing or the encoding is unknown.
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise errors.ConfigError(
                "tiktoken is required: pip install 'ceng-context[tokenize]'"
            ) from exc
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as exc:
            raise errors.ConfigError(
                f"unknown tiktoken encoding {encoding_name!r}"
            ) from exc
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(tokens[:max_tokens])


def default_tokenizer() -> Tokenizer:
    """Returns tiktoken if installed, else the heuristic estimator."""
    try:
        return TiktokenTokenizer()
    except errors.ConfigError:
        return HeuristicTokenizer()


def for_model(model: str) -> Tokenizer:
    """Returns the best available tokenizer for `model`.

    Uses the tiktoken encoding registered for the model when tiktoken is
    installed and knows the model; otherwise falls back to
    `default_tokenizer()` (tiktoken's `cl100k_base`, or the heuristic).

    Args:
        model: Model identifier, e.g. "gpt-4o-mini".
    """
    try:
        import tiktoken
    except ImportError:
        return HeuristicTokenizer()
    try:
        return TiktokenTokenizer(tiktoken.encoding_for_model(model).name)
    except KeyError:
        return default_tokenizer()
