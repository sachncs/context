"""Query expansion: search with the wording the document might use.

A question and its evidence often use different words ("capital expenditure"
against "purchases of property, plant and equipment"). One cheap model call
proposes alternative search phrasings; their rankings are fused with the
original query's by reciprocal rank, so a miss on one wording can be caught by
another. Failures never block a question: without rewrites the original query
is used alone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from foveate import errors, prompts
from foveate import messages as messages_lib
from foveate import runtime as runtime_lib
from foveate.internals import jsonout
from foveate.selection import base

logger = logging.getLogger("foveate.selection")
RRF_K = 60
EXPAND = prompts.PromptTemplate(
    name="selection.expand",
    version="1",
    system=(
        "You rewrite search queries. You never answer the question and you "
        "never invent facts."
    ),
    user=(
        "Write {count} different search queries that would find the passage "
        "answering this question in a long document. Use the wording the "
        "document itself is likely to use: synonyms, formal line-item or "
        "section names, abbreviations. Keep company names, periods and "
        "numbers. Reply with JSON only: "
        '{{"queries": ["...", "..."]}}\n\nQuestion: {question}'
    ),
)


async def rewrites(
    runtime: runtime_lib.Runtime, question: str, count: int
) -> list[str]:
    """Returns up to `count` alternative phrasings (empty if the call fails)."""
    try:
        reply = await runtime.complete(
            (
                messages_lib.Message(messages_lib.Role.SYSTEM, EXPAND.system),
                messages_lib.Message(
                    messages_lib.Role.USER,
                    EXPAND.render_user(question=question, count=str(count)),
                ),
            ),
            source="expand",
            namespace=f"expand:{EXPAND.fingerprint}",
            max_tokens=400,
        )
        queries = jsonout.extract_json(reply.text).get("queries")
    except errors.FoveateError as exc:
        logger.warning("query expansion failed: %s", exc)
        return []
    if not isinstance(queries, list):
        return []
    seen = {question.strip().casefold()}
    out = []
    for query in queries:
        text = str(query).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out[:count]


def fuse(rankings: Sequence[Sequence[base.Hit]], k: int) -> list[base.Hit]:
    """Fuses several chunk rankings by reciprocal rank; returns the top `k`."""
    scores: dict[str, float] = {}
    chunks = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            chunks[hit.chunk.id] = hit.chunk
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (
                RRF_K + rank
            )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [base.Hit(chunks[i], s) for i, s in ordered[:k]]
