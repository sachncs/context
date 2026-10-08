"""Choosing which chunks and pages go into the context window."""

from __future__ import annotations

from collections.abc import Sequence

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.backends import embeddings
from foveate.documents import chunking
from foveate.selection.base import Hit, Retriever, page_scores, ranked_pages
from foveate.selection.bm25 import BM25Retriever
from foveate.selection.embedding import EmbeddingRetriever
from foveate.selection.hybrid import HybridRetriever
from foveate.selection.rerank import LlmReranker

METHODS = ("bm25", "embedding", "hybrid")


def build_retriever(
    method: str,
    chunks: Sequence[chunking.Chunk],
    runtime: runtime_lib.Runtime,
    rerank: bool = False,
    candidates: int = 20,
) -> Retriever:
    """Builds a retriever by name.

    Args:
        method: `bm25` (lexical, no dependencies), `embedding` (needs
            `runtime.embedder`), or `hybrid` (BM25 plus embeddings fused by
            RRF; falls back to BM25 alone if the runtime has no embedder).
        chunks: Chunks to index.
        runtime: Supplies the embedder, cache and (for re-ranking) the model.
        rerank: Re-rank the top candidates with the model.
        candidates: Candidates handed to the re-ranker.

    Raises:
        ConfigError: For unknown methods or `embedding` without an embedder.
    """
    if method not in METHODS:
        raise errors.ConfigError(
            f"unknown retrieval method {method!r}; known: {', '.join(METHODS)}"
        )
    embedder: embeddings.Embedder | None = runtime.embedder
    lexical = BM25Retriever(chunks)
    if method == "bm25" or (method == "hybrid" and embedder is None):
        retriever: Retriever = lexical
    else:
        if embedder is None:
            raise errors.ConfigError(
                "embedding retrieval needs Runtime(embedder=...)"
            )
        dense = EmbeddingRetriever(chunks, embedder, runtime.cache)
        retriever = (
            dense
            if method == "embedding"
            else HybridRetriever([lexical, dense])
        )
    if rerank:
        retriever = LlmReranker(retriever, runtime, candidates)
    return retriever


__all__ = [
    "METHODS",
    "BM25Retriever",
    "EmbeddingRetriever",
    "Hit",
    "HybridRetriever",
    "LlmReranker",
    "Retriever",
    "build_retriever",
    "page_scores",
    "ranked_pages",
]
