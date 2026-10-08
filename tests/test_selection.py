import asyncio
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveate import errors
from foveate import runtime as runtime_lib
from foveate.backends import embeddings
from foveate.cache import base as cache_base
from foveate.documents import Document, chunking
from foveate.documents import page as page_lib
from foveate.selection import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    LlmReranker,
    build_retriever,
    page_scores,
    ranked_pages,
)
from foveate.tokenizers import HeuristicTokenizer
from tests import faults

TOK = HeuristicTokenizer()
TOPICS = {
    1: "The company reported record revenue growth in the cloud segment.",
    2: "Capital expenditures were 1577 million for purchases of property plant and equipment.",
    3: "Employee headcount rose and the cafeteria menu changed seasonally.",
    4: "The board approved a dividend of 1.50 per share payable in March.",
    5: "Legal proceedings include a patent dispute settled for an undisclosed sum.",
}


def corpus():
    pages = tuple(
        page_lib.Page(n, text, TOK.count(text), (f"SECTION {n}",))
        for n, text in TOPICS.items()
    )
    return [Document("filing", pages)]


def chunks():
    return chunking.chunk_all(corpus(), TOK)


def run(coro):
    return asyncio.run(coro)


class TestChunking:
    def test_chunks_carry_page_and_heading(self):
        cs = chunks()
        assert [c.page for c in cs] == [1, 2, 3, 4, 5]
        assert cs[1].heading == "SECTION 2" and cs[1].id == "filing:2:0"
        assert cs[1].searchable.startswith("SECTION 2\n")

    def test_long_page_is_subdivided_and_blank_pages_skipped(self):
        text = "Sentence number %d is here. " * 1
        long_text = "".join(
            f"Sentence number {i} is here. " for i in range(200)
        )
        doc = Document(
            "d",
            (
                page_lib.Page(1, long_text, TOK.count(long_text)),
                page_lib.Page(2, "  ", 1),
            ),
        )
        cs = chunking.PageChunker(max_tokens=100).chunk(doc, TOK)
        assert len(cs) > 3 and {c.page for c in cs} == {1}
        assert all(c.tokens <= 100 for c in cs) and text

    def test_validation(self):
        with pytest.raises(errors.ConfigError):
            chunking.PageChunker(0)
        with pytest.raises(errors.ValidationError):
            chunking.chunk_all(corpus() + corpus(), TOK)


class TestBM25:
    def test_finds_the_right_page(self):
        r = BM25Retriever(chunks())
        top = run(
            r.search(
                "what were capital expenditures for property and equipment", 3
            )
        )
        assert top[0].chunk.page == 2
        assert run(r.search("dividend per share", 1))[0].chunk.page == 4

    def test_no_match_returns_nothing(self):
        assert run(BM25Retriever(chunks()).search("zebra quantum", 5)) == []

    def test_heading_is_searchable(self):
        assert (
            run(BM25Retriever(chunks()).search("SECTION 3", 1))[0].chunk.page
            == 3
        )

    @settings(max_examples=40, deadline=None)
    @given(st.text(alphabet="abcdefghij ", max_size=30), st.integers(1, 6))
    def test_results_sorted_and_bounded(self, query, k):
        hits = run(BM25Retriever(chunks()).search(query, k))
        scores = [h.score for h in hits]
        assert len(hits) <= k and scores == sorted(scores, reverse=True)
        assert all(s > 0 for s in scores)

    def test_validation(self):
        for kw in ({"k1": 0}, {"b": 2}):
            with pytest.raises(errors.ConfigError):
                BM25Retriever(chunks(), **kw)


class CountingEmbedder(embeddings.HashingEmbedder):
    """Real hashing embedder that records how much work it did."""

    def __init__(self):
        super().__init__(256)
        self.texts = 0

    async def embed(self, texts, kind=embeddings.PASSAGE):
        self.texts += len(texts)
        return await super().embed(texts, kind)


class TestEmbeddingAndHybrid:
    def test_hashing_embedder_is_unit_length_and_deterministic(self):
        e = embeddings.HashingEmbedder(64)
        (a,) = run(e.embed(["capital expenditures"]))
        (b,) = run(e.embed(["capital expenditures"]))
        assert a == b and sum(x * x for x in a) == pytest.approx(1.0)
        assert run(e.embed([""]))[0] == [0.0] * 64
        with pytest.raises(errors.ConfigError):
            embeddings.HashingEmbedder(2)

    def test_dense_search_and_vector_cache(self):
        cache = cache_base.MemoryCache()
        e = CountingEmbedder()
        r = EmbeddingRetriever(chunks(), e, cache)
        assert run(r.search("dividend payable in March", 1))[0].chunk.page == 4
        first = e.texts
        again = EmbeddingRetriever(chunks(), e, cache)
        run(again.search("dividend payable in March", 1))
        assert e.texts == first + 1  # only the new query was embedded

    def test_corrupt_cached_vector_is_recomputed(self):
        cache = cache_base.MemoryCache()
        r = EmbeddingRetriever(chunks(), embeddings.HashingEmbedder(64), cache)
        run(r.search("revenue", 1))
        for key in list(cache.data):
            cache.data[key] = "{not json"
        again = EmbeddingRetriever(
            chunks(), embeddings.HashingEmbedder(64), cache
        )
        assert run(again.search("revenue", 1))

    def test_hybrid_fusion(self):
        lexical = BM25Retriever(chunks())
        dense = EmbeddingRetriever(chunks(), embeddings.HashingEmbedder(256))
        hybrid = HybridRetriever([lexical, dense], weights=[2, 1])
        hits = run(hybrid.search("patent dispute settlement", 2))
        assert hits[0].chunk.page == 5 and len(hits) <= 2
        for bad in (
            {"retrievers": []},
            {"weights": [1]},
            {"weights": [1, -1]},
            {"rrf_k": 0},
        ):
            kw = {"retrievers": [lexical, dense], **bad}
            with pytest.raises(errors.ConfigError):
                HybridRetriever(**kw)


class TestRerank:
    def runtime(self, reply):
        backend = faults.ScriptedBackend(
            reply if isinstance(reply, list) else [reply]
        )
        return runtime_lib.Runtime(
            backend=backend, cache=cache_base.MemoryCache()
        ), backend

    def test_model_ordering_is_applied(self):
        rt, _ = self.runtime(
            json.dumps({"ranking": ["filing:5:0", "filing:2:0"]})
        )
        r = LlmReranker(BM25Retriever(chunks()), rt, candidates=5)
        hits = run(r.search("company legal revenue property", 3))
        assert [h.chunk.page for h in hits][:2] == [5, 2]

    @pytest.mark.parametrize(
        "reply", ["not json", json.dumps({"ranking": "bad"})]
    )
    def test_bad_reply_keeps_first_stage_order(self, reply):
        rt, _ = self.runtime(reply)
        base_hits = run(
            BM25Retriever(chunks()).search("company revenue legal", 3)
        )
        hits = run(
            LlmReranker(BM25Retriever(chunks()), rt).search(
                "company revenue legal", 3
            )
        )
        assert [h.chunk.id for h in hits] == [h.chunk.id for h in base_hits]

    def test_backend_failure_keeps_first_stage_order(self):
        rt, _ = self.runtime(errors.PermanentBackendError("down"))
        hits = run(
            LlmReranker(BM25Retriever(chunks()), rt).search("dividend", 2)
        )
        assert hits and hits[0].chunk.page == 4

    def test_validation_and_single_candidate(self):
        rt, backend = self.runtime("x")
        with pytest.raises(errors.ConfigError):
            LlmReranker(BM25Retriever(chunks()), rt, candidates=1)
        one = run(
            LlmReranker(BM25Retriever(chunks()), rt).search("zebra dividend", 3)
        )
        assert len(one) == 1 and backend.requests == []


class TestFactoryAndPages:
    def rt(self, **kw):
        return runtime_lib.Runtime(backend=faults.ScriptedBackend(), **kw)

    def test_methods(self):
        assert isinstance(
            build_retriever("bm25", chunks(), self.rt()), BM25Retriever
        )
        assert isinstance(
            build_retriever("hybrid", chunks(), self.rt()), BM25Retriever
        )
        rt = self.rt(embedder=embeddings.HashingEmbedder(64))
        assert isinstance(
            build_retriever("hybrid", chunks(), rt), HybridRetriever
        )
        assert isinstance(
            build_retriever("embedding", chunks(), rt), EmbeddingRetriever
        )
        assert isinstance(
            build_retriever("bm25", chunks(), rt, rerank=True), LlmReranker
        )
        for bad in ("nope",):
            with pytest.raises(errors.ConfigError):
                build_retriever(bad, chunks(), rt)
        with pytest.raises(errors.ConfigError):
            build_retriever("embedding", chunks(), self.rt())

    def test_page_scores_reward_multiple_matching_chunks(self):
        c = chunks()
        hits = [
            type(run(BM25Retriever(c).search("revenue", 1))[0])(c[0], 1.0),
            type(run(BM25Retriever(c).search("revenue", 1))[0])(c[1], 1.0),
        ]
        scores = page_scores(hits)
        assert scores[("filing", 1)] == scores[("filing", 2)] == 1.0
        two = [*hits, type(hits[0])(c[0], 0.5)]
        assert page_scores(two)[("filing", 1)] == pytest.approx(1.05)
        assert ranked_pages(two)[0][0] == ("filing", 1)
