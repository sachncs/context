"""ceng — Partition-Prompt-Aggregate context engineering for LLMs.

Public surface:

* :func:`ppa_compress` — compress a long context by aggregating leaf summaries.
* :func:`ppa_check` — run the paper's self-consistency probe.
* :func:`set_backend` — pick the LLM backend at runtime.

Default backend is ``"litellm"`` which transparently supports vLLM (via an
OpenAI-compatible server), OpenAI, Anthropic, and any other provider
``litellm`` speaks.
"""

from ceng.backends import (
    Backend,
    LiteLLMBackend,
    OpenAIBackend,
    VLLMBackend,
    available_backends,
    get_backend,
    set_backend,
)
from ceng.cache import Cache
from ceng.check import Verdict, ppa_check
from ceng.compress import ppa_compress
from ceng.partition import Partition, partition_text
from ceng.tokens import count_tokens, token_budget_split

__all__ = [
    "Backend",
    "LiteLLMBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "available_backends",
    "get_backend",
    "set_backend",
    "Cache",
    "Verdict",
    "ppa_compress",
    "ppa_check",
    "Partition",
    "partition_text",
    "count_tokens",
    "token_budget_split",
]

__version__ = "0.1.0"
