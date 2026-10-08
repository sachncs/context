---
title: "4. Keep prefixes stable"
description: "How provider prompt caching works, why a changing prefix makes it miss, and how to structure prompts so it hits."
---

**Rule: put the parts of the prompt that never change at the start, in the same order, byte for byte;
put what varies at the end.**

## The problem

When a provider sees a prompt that starts with the same text as a recent one, it can skip recomputing
that part and charge a lower rate for it. This is **prompt caching**. The saving is large: the Manus
team reports that with Claude Sonnet, cached input tokens cost $0.30 per million against $3 for
uncached, a tenfold difference, and that their agents read about 100 input tokens for every token they
write. A 2026 evaluation of caching on long agent tasks measured 41% to 80% lower API cost and 13% to
31% faster first tokens.

The catch is that caching matches from the **start** of the prompt. One changed character early on, a
timestamp, a reordered list of tools, a different key order in serialised JSON, and everything after it
is a miss.

## What to do

* **Order by how often things change.** Fixed instructions and tool definitions first; slow-changing
  context next; the user's question and volatile data last.
* **Append, do not edit.** Add new turns to the end; do not rewrite earlier ones.
* **Serialise deterministically.** Many JSON libraries do not guarantee key order. Sort keys, or use a
  fixed schema.
* **Keep dynamic content out of the system prompt.** The 2026 evaluation found that placing dynamic
  content at the end of the system prompt and excluding dynamic tool results gave more consistent savings
  than caching everything, which could even add latency.
* **Watch the cache hit rate, not the token count.** Providers report cached tokens in the response.

A caution from the compression research: a policy that cuts tokens but rewrites old turns can break the
cache and cost *more*. In *Beyond Token Savings*, policies using 0.56 to 0.63 times the tokens cost 0.71
to 0.95 times as much as full context once caching was billed.

## In Foveate

* The answer prompt puts the stable instructions and outline first and the question and any retry
  feedback last.
* `Answer.usage.cached_tokens` shows how many prompt tokens the provider served from cache.
* Add a cached rate to your `PriceTable` (`Price(..., cached_per_million=...)`) so `cost_usd` reflects it.
* `HistoryCompressor(trigger=0.8, target=0.5)` compresses in occasional larger steps instead of changing
  the history a little every turn, which keeps the prefix stable longer.

## Mistakes to avoid

* **Putting the current time or a request id at the top** of the prompt.
* **Rebuilding the tool list in a different order** each run.
* **Compressing a little every turn.** Prefer fewer, larger compressions.
