"""ceng — Partition-Prompt-Aggregate context engineering for LLMs.

Public surface:

* :func:`ppa_compress` — compress a long context by partitioning,
  summarising each chunk, and aggregating the summaries.
* :func:`compress_with_stats` / :class:`CompressResult` — same as
  :func:`ppa_compress` but with cache-hit bookkeeping.
* :func:`compress_to_bundle` / :class:`CompressionBundle` — same as
  :func:`ppa_compress` but with per-leaf provenance.
* :func:`ppa_compress_to_okf` — run the compressor and persist the
  result as an Open Knowledge Format bundle on disk.
* :func:`ppa_check` — run the paper's binary-tree self-consistency
  probe to detect the macro fallacy.
* :func:`set_backend` / :func:`get_backend` / :func:`reset_backend` —
  pick the LLM backend (``"litellm"``, ``"vllm"``, or ``"openai"``).

The :mod:`ceng.okf` module exposes the primitives for the OKF format
directly (:class:`ceng.okf.Concept`, :class:`ceng.okf.Frontmatter`,
:func:`ceng.okf.read_bundle`, :func:`ceng.okf.write_bundle`).
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
from ceng.compress import (
    CompressResult,
    CompressionBundle,
    LeafArtifact,
    compress_to_bundle,
    compress_with_stats,
    ppa_compress,
    ppa_compress_to_okf,
)
from ceng.okf import (
    CENG_BUNDLE_INDEX,
    CENG_COMBINED_SUMMARY,
    CENG_LEAF_SUMMARY,
    OKF_VERSION,
    RESERVED_INDEX,
    RESERVED_LOG,
    Concept,
    Frontmatter,
    cross_links,
    find_concept,
    now_iso,
    parse_concept,
    parse_frontmatter,
    read_bundle,
    read_concept_file,
    render_concept,
    render_frontmatter,
    write_bundle,
    write_concept_file,
)
from ceng.partition import Partition, partition_text
from ceng.tokens import count_tokens, token_budget_split

__all__ = [
    # backends
    "Backend",
    "LiteLLMBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "available_backends",
    "get_backend",
    "reset_backend",
    "set_backend",
    # cache
    "Cache",
    "make_key",
    "NAMESPACE_PPA_CHECK",
    "NAMESPACE_SUMMARIZE",
    # consistency probe
    "Verdict",
    "LeafEstimate",
    "ppa_check",
    # compression
    "CompressResult",
    "CompressionBundle",
    "LeafArtifact",
    "compress_with_stats",
    "compress_to_bundle",
    "ppa_compress",
    "ppa_compress_to_okf",
    # OKF primitives
    "Concept",
    "Frontmatter",
    "now_iso",
    "parse_concept",
    "parse_frontmatter",
    "render_concept",
    "render_frontmatter",
    "read_bundle",
    "read_concept_file",
    "write_bundle",
    "write_concept_file",
    "cross_links",
    "find_concept",
    "OKF_VERSION",
    "RESERVED_INDEX",
    "RESERVED_LOG",
    "CENG_BUNDLE_INDEX",
    "CENG_COMBINED_SUMMARY",
    "CENG_LEAF_SUMMARY",
    # partitioning + tokens
    "Partition",
    "partition_text",
    "count_tokens",
    "token_budget_split",
]

__version__ = "0.2.0"
