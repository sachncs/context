"""ceng — Partition-Prompt-Aggregate context engineering for LLMs.

Public surface:

* :func:`ppa_compress` — compress a long context by partitioning,
  summarising each chunk, and aggregating the summaries.
* :func:`ppa_check` — run the paper's binary-tree self-consistency
  probe to detect the macro fallacy in an LLM's population-level
  estimates.
* :func:`compress_with_stats` / :class:`CompressResult` — same as
  :func:`ppa_compress` but with cache-hit bookkeeping returned.
* :func:`set_backend` / :func:`get_backend` / :func:`reset_backend` —
  pick the LLM backend (``"litellm"``, ``"vllm"``, or ``"openai"``) at
  runtime.

Default backend is :class:`LiteLLMBackend`, which transparently
supports vLLM (via an OpenAI-compatible server), OpenAI, Anthropic,
and any other provider ``litellm`` speaks.
"""

from ceng.backends import (
    Backend,
    LiteLLMBackend,
    OpenAIBackend,
    VLLMBackend,
    available_backends,
    get_backend,
    reset_backend,
    set_backend,
)
from ceng.cache import (
    NAMESPACE_PPA_CHECK,
    NAMESPACE_SUMMARIZE,
    Cache,
    make_key,
)
from ceng.check import LeafEstimate, Verdict, ppa_check
from ceng.compress import CompressResult, compress_with_stats, ppa_compress
from ceng.partition import Partition, partition_text
from ceng.tokens import count_tokens, token_budget_split

__all__ = [
    "Backend",
    "LiteLLMBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "available_backends",
    "get_backend",
    "reset_backend",
    "set_backend",
    "Cache",
    "make_key",
    "NAMESPACE_PPA_CHECK",
    "NAMESPACE_SUMMARIZE",
    "Verdict",
    "LeafEstimate",
    "ppa_check",
    "CompressResult",
    "compress_with_stats",
    "ppa_compress",
    "Partition",
    "partition_text",
    "count_tokens",
    "token_budget_split",
]

__version__ = "0.1.0"
