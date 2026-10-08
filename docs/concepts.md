# Concepts

## Pages are the unit

Everything is addressed by `(document id, page number)`. A `Document` is an
immutable tuple of `Page(number, text, tokens, headings)`. PDFs keep their real
pages; formats without pages (text, Markdown, HTML, DOCX) are split into
pseudo-pages of about 500 tokens. Citations, retrieval hits, the outline and
tool calls all use the same page numbers, so you can always open the page.

## Foveation: three levels of detail

Given a question and a token budget, Foveate decides how much of each page the
model sees:

| Tier | Meaning | Cost |
|---|---|---|
| FULL (fovea) | The page text, verbatim. Best-scoring pages and their closest neighbours. | full |
| CONDENSED (parafovea) | Extractive summary of pages next to the fovea, so context is not lost. | reduced |
| OUTLINE (periphery) | One line: page number and heading. The model can see the document exists and ask for more. | a few tokens |
| DROPPED | Not shown. | 0 |

The allocation never exceeds the budget. It is a pure function of the scores,
so `Plan` can show it before any model call.

## Selection

A `Retriever` ranks chunks (page slices that remember their page and heading):

* `bm25` (default): pure Python, no dependencies, strong on numbers and names.
* `embedding`: any OpenAI-compatible `/embeddings` endpoint.
* `hybrid`: reciprocal-rank fusion of BM25 and embeddings (falls back to BM25
  when no embedder is configured).
* `rerank=True`: the model re-scores the candidates.

Chunk scores are folded into page scores: a page scores its best chunk plus a
tenth of each further matching chunk.

## Grounded answers

The model must answer in JSON:

```json
{"found": true, "answer": "...",
 "citations": [{"doc": "acme_2018", "page": 7, "quote": "exact words from the page"}]}
```

Foveate then checks each quote **deterministically** against the cited page
(normalised for whitespace, case and punctuation; a long common subsequence also
counts). The result is `Answer.grounded`, true only if there is at least one
citation and all are verified. If an answer is unsupported, Foveate retries
once with the problem described and, if the model asked for pages it did not
see, with those pages added. After that the answer is returned with
`grounded=False` (`ungrounded="flag"`) or replaced by an explicit not-found
message (`ungrounded="abstain"`).

Abstention is a feature: when the document does not contain the answer, "not
found" is correct and a confident guess is a hallucination.

## Plans

`Foveator.plan(question, documents)` returns the page allocation, document
tokens, expected prompt size and estimated cost without calling the model. Use
it to size budgets, show users what will be sent, or gate expensive calls.

## Compression of conversations and tool output

Long chat histories and tool results are handled by `Context.compress`:

```python
smaller = Context(messages, runtime).compress("ushape+ppa", budget=4000)
```

Strategies are registered classes; `a+b` runs them in sequence and `a|b` falls
back to `b` if `a` fails. `tool_output` shrinks JSON, CSV, HTML and logs
without a model while keeping their shape (keys, header, counts of what was
cut). Every result carries a `CompressionReport` with tokens before and after,
steps, cost and whether anything was truncated.

## The runtime

`Runtime` is the one object that talks to a model: backend, cache, tokenizer,
observers, prices and options. Everything funnels through `Runtime.complete`,
which adds caching, single-flight de-duplication of identical in-flight calls,
and recovery when a reasoning model spends its whole token cap thinking.
Wrap any backend in `ResilientBackend` for retry with jitter, timeouts, a
circuit breaker, concurrency and rate limits, and a total deadline.

## Prompt-injection fencing

Retrieved pages are data. They are placed inside a fence the model is told never
to obey, and `foveate.fencing.escape` stops page text from closing the fence or
faking a page marker. `fencing.suspicious(text)` flags the usual "ignore all
previous instructions" phrasing so you can log or drop such pages. This removes
the cheap attacks; it is not a complete defence.
