"""The systems under test: how a long document reaches the model.

Every pipeline uses the same answer prompt, JSON schema and citation check
(`Foveator`), so the only difference between them is *how evidence is chosen
and assembled* and whether the answer is verified and retried.
"""

from __future__ import annotations

import abc
import dataclasses
import time
from typing import ClassVar

from foveate import Foveator, errors, models
from foveate import runtime as runtime_lib
from foveate.documents import Document, chunking
from foveate.foveation import Foveation, PagePlan, Tier
from foveate.grounding import NOT_FOUND
from foveate.internals import registry
from foveate.selection import base as selection_base
from foveate.selection import bm25

OVERFLOW_MARGIN = 800


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """What a pipeline produced for one question.

    Attributes:
        text: The answer text (or the not-found message).
        found: Whether the model said it found the answer.
        abstained: Whether the pipeline declined to answer.
        cited_pages: Pages of the document that were cited.
        verified: Cited quotes that exist on their cited page.
        citations: Total citations made.
        grounded: Foveate's grounded flag (citations present and verified).
        prompt_tokens: Provider-reported prompt tokens over all rounds.
        completion_tokens: Provider-reported completion tokens.
        seconds: Wall-clock latency.
        rounds: Model rounds used.
        error: Failure description; empty on success.
        overflow: The document does not fit the model window (no call made).
    """

    text: str = ""
    found: bool = False
    abstained: bool = False
    cited_pages: tuple[int, ...] = ()
    verified: int = 0
    citations: int = 0
    grounded: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    rounds: int = 0
    error: str = ""
    overflow: bool = False


class Pipeline(abc.ABC):
    """A way of answering a question about one document."""

    registry: ClassVar[registry.Registry[type[Pipeline]]] = registry.Registry(
        "pipeline"
    )
    name: ClassVar[str] = ""

    def __init__(self, runtime: runtime_lib.Runtime, budget: int) -> None:
        """Creates the pipeline.

        Args:
            runtime: Model access.
            budget: Prompt token budget for evidence-limited pipelines.
        """
        self.runtime = runtime
        self.budget = budget

    @abc.abstractmethod
    async def answer(
        self, question: str, document: Document, tag: str = ""
    ) -> Result:
        """Answers `question` about `document`.

        Args:
            question: The question.
            document: The (possibly needle-modified) document.
            tag: Run tag; different tags are independent samples.
        """

    def foveator(self, tag: str, rounds: int = 1) -> Foveator:
        """Returns a Foveator configured for this pipeline's sampling."""
        sampled = tag != ""
        return Foveator(
            self.runtime,
            budget=self.budget,
            max_rounds=rounds,
            temperature=0.7 if sampled else 0.0,
            run_tag=tag,
        )

    async def run_fixed(
        self,
        question: str,
        document: Document,
        fov: Foveation,
        tag: str,
    ) -> Result:
        """Asks once with a given page allocation and scores the reply."""
        started = time.monotonic()
        foveator = self.foveator(tag)
        try:
            parsed, raw, usage = await foveator.ask_once(question, fov)
        except errors.FoveateError as exc:
            return Result(error=f"{type(exc).__name__}: {exc}"[:200])
        del raw
        citations = foveator.verify(parsed, [document])
        found = bool(parsed.get("found"))
        text = str(parsed.get("answer") or "").strip()
        abstained = not found or not text
        return Result(
            text=NOT_FOUND if abstained else text,
            found=found and not abstained,
            abstained=abstained,
            cited_pages=tuple(
                dict.fromkeys(
                    c.page for c in citations if c.doc_id == document.id
                )
            ),
            verified=sum(c.verified for c in citations),
            citations=len(citations),
            grounded=bool(citations) and all(c.verified for c in citations),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            seconds=time.monotonic() - started,
            rounds=1,
        )


def pages_in_full(
    document: Document, numbers: list[int], tokenizer_count: int | None = None
) -> Foveation:
    """Builds a foveation showing `numbers` in full and dropping the rest."""
    chosen = set(numbers)
    plans = []
    for page in document.pages:
        if page.number in chosen:
            plans.append(
                PagePlan(
                    document.id,
                    page.number,
                    Tier.FULL,
                    0.0,
                    page.tokens,
                    page.tokens,
                    page.text,
                )
            )
        else:
            plans.append(
                PagePlan(
                    document.id,
                    page.number,
                    Tier.DROPPED,
                    0.0,
                    page.tokens,
                    0,
                    "",
                )
            )
    return Foveation(
        tuple(plans), tokenizer_count or sum(p.tokens for p in document.pages)
    )


@Pipeline.registry.register("full-context")
class FullContext(Pipeline):
    """Send the whole document (what most people do first).

    If the document does not fit the model window no call is made and the
    result is marked `overflow`: a real provider would reject it.
    """

    name = "full-context"

    async def answer(
        self, question: str, document: Document, tag: str = ""
    ) -> Result:
        info = models.lookup(self.runtime.model)
        window = self.runtime.context_window or (
            info.context_window if info else None
        )
        if (
            window is not None
            and document.token_count + OVERFLOW_MARGIN > window
        ):
            return Result(
                error=(
                    f"document is {document.token_count} tokens; "
                    f"window is {window}"
                ),
                overflow=True,
            )
        return await self.run_fixed(
            question, document, pages_in_full(document, document.numbers), tag
        )


@Pipeline.registry.register("truncate")
class NaiveTruncate(Pipeline):
    """Keep the first pages that fit the budget (silent truncation)."""

    name = "truncate"

    async def answer(
        self, question: str, document: Document, tag: str = ""
    ) -> Result:
        remaining = self.budget - 1200
        kept = []
        for page in document.pages:
            if page.tokens > remaining:
                break
            kept.append(page.number)
            remaining -= page.tokens
        return await self.run_fixed(
            question, document, pages_in_full(document, kept), tag
        )


@Pipeline.registry.register("naive-rag")
class NaiveRag(Pipeline):
    """Fixed-size chunks, BM25 top-k up to the budget, one shot, no checks."""

    name = "naive-rag"

    async def answer(
        self, question: str, document: Document, tag: str = ""
    ) -> Result:
        tokenizer = self.runtime.tokenizer
        chunks = chunking.PageChunker(256).chunk(document, tokenizer)
        hits = await bm25.BM25Retriever(chunks).search(question, 200)
        selected = chunk_foveation(document, hits, self.budget - 1200)
        return await self.run_fixed(question, document, selected, tag)


def chunk_foveation(
    document: Document, hits: list[selection_base.Hit], budget: int
) -> Foveation:
    """Shows only the retrieved chunks (not whole pages), best first."""
    by_page: dict[int, list[str]] = {}
    remaining = budget
    for hit in hits:
        if hit.chunk.tokens > remaining:
            continue
        by_page.setdefault(hit.chunk.page, []).append(hit.chunk.text)
        remaining -= hit.chunk.tokens
    plans = []
    for page in document.pages:
        text = "\n...\n".join(by_page.get(page.number, []))
        tier = Tier.FULL if text else Tier.DROPPED
        plans.append(
            PagePlan(document.id, page.number, tier, 0.0, page.tokens, 0, text)
        )
    return Foveation(tuple(plans), budget)


@Pipeline.registry.register("foveate")
class Foveate(Pipeline):
    """The full pipeline: retrieve, foveate, answer, verify, retry."""

    name = "foveate"

    async def answer(
        self, question: str, document: Document, tag: str = ""
    ) -> Result:
        foveator = self.foveator(tag, rounds=2)
        try:
            answer = await foveator.aask(question, [document])
        except errors.FoveateError as exc:
            return Result(error=f"{type(exc).__name__}: {exc}"[:200])
        return Result(
            text=answer.text,
            found=answer.found,
            abstained=answer.abstained,
            cited_pages=tuple(
                dict.fromkeys(
                    c.page for c in answer.citations if c.doc_id == document.id
                )
            ),
            verified=sum(c.verified for c in answer.citations),
            citations=len(answer.citations),
            grounded=answer.grounded,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
            seconds=answer.seconds,
            rounds=answer.rounds,
        )
