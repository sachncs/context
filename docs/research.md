---
title: "Research and sources"
description: "The papers, playbooks and engineering write-ups behind Foveate, with what each one found and what Foveate takes from it."
---

Context engineering is the work of deciding what a model sees: which text, in what
order, at what level of detail, remembered from where, and checked how. It is wider
than prompt writing and wider than compression. This page lists the sources Foveate is
built on. Quoted wording and numbers were read on the source page on 8 October 2026.
Sources we have not read in full are listed by title and link only, with no claims
attached.

## Definitions and surveys

* Hua et al. (2025), *Context Engineering 2.0: The Context of Context Engineering*,
  [arXiv:2510.26493](https://arxiv.org/abs/2510.26493). **Foveate takes** the scope: selection,
  structure, compression, memory and verification are one pipeline.
* *A Survey of Context Engineering for Large Language Models*,
  [arXiv:2507.13334](https://arxiv.org/abs/2507.13334). A map of the field.
* *Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness
  Engineering*, [arXiv:2604.08224](https://arxiv.org/abs/2604.08224).

## Why long contexts fail

* Xia et al. (2026), *Diagnosing and Mitigating Context Rot in Long-horizon Search*,
  [arXiv:2606.29718](https://arxiv.org/abs/2606.29718). Models "give up or provide
  uncertain incorrect answers long before exhausting the context window", and the rate
  rises with context length. **Foveate takes** small budgets by default and explicit
  abstention instead of a forced answer.
* Anthropic (2025), [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
  "As the number of tokens in the context window increases, the model's ability to
  accurately recall information from that context decreases."
* *Classifier Context Rot*, [arXiv:2605.12366](https://arxiv.org/abs/2605.12366), and
  *Intelligence Degradation in Long-Context LLMs*,
  [arXiv:2601.15300](https://arxiv.org/abs/2601.15300): further evidence of degradation with length.
* Zeng et al. (2026), *LOCA-bench*, [arXiv:2602.07962](https://arxiv.org/abs/2602.07962).
  Agent performance "generally degrades as the environment states grow more complex",
  and context management techniques substantially reduce the decline.

## Needle-in-a-haystack and long-context evaluation

* Modarressi et al. (2025), *NoLiMa: Long-Context Evaluation Beyond Literal Matching*,
  [arXiv:2502.05167](https://arxiv.org/abs/2502.05167). Needles and questions share almost no
  words; "at 32K tokens, 11 of 13 tested models fell below 50% of their baseline
  performance" (revised abstract). **Foveate takes** non-literal needles for the
  [needle benchmark](benchmarks.md) and the lesson that keyword retrieval alone is not enough.
* Kuratov et al., *BABILong*, [arXiv:2406.10149](https://arxiv.org/abs/2406.10149); Bai et al.,
  *LongBench v2*; Hsieh et al., *RULER* (2024). The standard long-context suites; Foveate's
  multi-needle generator follows the RULER idea.
* *Sequential-NIAH*, [arXiv:2504.04713](https://arxiv.org/abs/2504.04713): needles that must
  be extracted in order.

## Compression

* *End-to-End Context Compression at Scale*,
  [arXiv:2606.09659](https://arxiv.org/abs/2606.09659);
  *AdmTree*, [arXiv:2512.04550](https://arxiv.org/abs/2512.04550);
  *Efficient Prompt Compression with Evaluator Heads*,
  [arXiv:2501.12959](https://arxiv.org/abs/2501.12959). Learned compressors and attention-based
  pruning. Foveate is the model-agnostic text layer that works with any API model.
* Wolf et al. (2026), *Partition, Prompt, Aggregate: Statistical Self-Consistency in Language
  Models*, [arXiv:2607.15277](https://arxiv.org/abs/2607.15277). The method behind Foveate's `ppa` strategy.
* Kang et al. (2026), *ACON: Optimizing Context Compression for Long-horizon LLM Agents*,
  [arXiv:2510.00615](https://arxiv.org/abs/2510.00615), ICML 2026. Reports 26-54% lower peak
  token use than compression baselines, and up to 46% better performance for smaller
  models. **Foveate takes** history and observation compression for agents; tuning the
  compression instructions from failures is on the [roadmap](roadmap.md).
* Lodha et al. (2026), *Less Context, Better Agents*,
  [arXiv:2606.10209](https://arxiv.org/abs/2606.10209). On 50 expense tasks, full history
  reached 71.0% completion with 1.48M tokens; pruning to the last five tool calls reached
  79.0% with 0.54M; pruning plus summaries reached 91.6% with 0.55M. **Foveate takes** the
  combination: keep recent tool results verbatim, summarise the rest.

## Query-aware pruning for retrieval

* Chirkova et al. (2025), *Provence: efficient and robust context pruning for
  retrieval-augmented generation*, ICLR 2025. Prunes per sentence, conditioned on the query.
* *AttentionRAG*, [arXiv:2503.10720](https://arxiv.org/abs/2503.10720);
  *EnComp*, [arXiv:2603.09222](https://arxiv.org/abs/2603.09222). **Foveate takes** the idea in
  the `query` compressor and the optional `query_aware` condensed tier: keep the sentence that
  matches the question, not the most frequent words.

## Caching and cost

* Lumer et al. (2026), *Don't Break the Cache: An Evaluation of Prompt Caching for
  Long-Horizon Agentic Tasks*, [arXiv:2601.06007](https://arxiv.org/abs/2601.06007). Prompt
  caching cut API cost by 41-80% and time to first token by 13-31%; placing dynamic content at the
  end of the system prompt and excluding dynamic tool results worked best. **Foveate takes**
  prompts whose stable part comes first and the question last, and reports `cached_tokens`.
* Manus (2025), [*Context Engineering for AI Agents: Lessons from Building Manus*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).
  Treats KV-cache hit rate as the most important production metric; with Claude Sonnet, cached
  input tokens cost 0.30 USD per million versus 3 USD uncached; keep context append-only with
  deterministic serialisation.

## Self-improving contexts and memory

* Zhang et al. (2026), *Agentic Context Engineering: Evolving Contexts for Self-Improving
  Language Models*, [arXiv:2510.04618](https://arxiv.org/abs/2510.04618), ICLR 2026. Contexts as
  evolving playbooks; reported gains of +10.6% on agents and +8.6% on finance. Implemented in
  `foveate.evolution`. Code: [ace-agent/ace](https://github.com/ace-agent/ace).
* Xu et al., *A-MEM: Agentic Memory for LLM Agents*, [arXiv:2502.12110](https://arxiv.org/abs/2502.12110);
  *Anatomy of Agentic Memory*, [arXiv:2602.19320](https://arxiv.org/abs/2602.19320);
  *A Survey of Agent Memory in the Second Half*, [arXiv:2602.06052](https://arxiv.org/abs/2602.06052).
  **Foveate takes** the split between working notes and durable facts, in `foveate.memory`.
* *Structured Context Engineering for File-Native Agentic Systems*,
  [arXiv:2602.05447](https://arxiv.org/abs/2602.05447): the format of context changes accuracy.

## Industry playbooks

* **Anthropic.** [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  (29 September 2025): compaction, tool result clearing ("one of the safest lightest touch
  forms of compaction"), structured note-taking, sub-agents, just-in-time retrieval. The
  [context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
  documentation describes tool result clearing with a trigger, a number of recent results to
  keep, and a placeholder so the model knows a result was removed. **Foveate takes**
  `clear_tool_results`, page tools for just-in-time retrieval, and `ppa` compaction.
* **OpenAI.** [*A practical guide to building agents*](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
  (2025): "maximize a single agent's capabilities first", build instructions from existing
  documents, guardrails as a "layered defense mechanism". The Agents SDK
  [session memory cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
  compares trimming (last N turns; lowest cost, hard cut-off) with summarisation (stronger
  long-range memory, risk of context distortion) and says to log summary prompts and outputs.
  **Foveate takes** trimming and summarising as separate, composable methods (`window`, `ushape`).
* **Google.** Milam and Gulli (November 2025), *Context Engineering: Sessions, Memory*
  (Kaggle and Google agents whitepaper): "The goal of context engineering is to ensure the
  model has no more and no less than the most relevant information to complete its task."
  Compaction strategies run from keeping the last N turns, through token-based truncation, to
  recursive summarisation, with count-based and time-based triggers, ideally computed
  asynchronously and persisted. Memory generation is extraction followed by consolidation.
  **Foveate takes** the session and long-term split in `foveate.memory`.
* **Amazon.** [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/amazon-bedrock-agent-core.html)
  provides managed short-term and long-term memory and OpenTelemetry-based observability;
  Strands Agents ship sliding-window and summarising conversation managers.
  **Foveate takes** the same two overflow policies and OpenTelemetry tracing.
* **Uber.** [*Enhanced Agentic-RAG*](https://www.uber.com/blog/enhanced-agentic-rag/)
  (29 May 2025): moving from plain RAG to pre-processing agents (query optimiser, source
  identifier), hybrid vector and BM25 retrieval, and a post-processor that de-duplicates chunks
  and "structures the context based on the positional order" gave "a relative 27%" more
  acceptable answers and a relative 60% reduction in incorrect advice. Uber's evaluation-first
  platform work argues for building evals on day one. **Foveate takes** hybrid retrieval,
  deduplicated, ordered context, and the [evaluation guide](guides/evaluating.md).

## Further reading supplied by the project owner

* Anthropic (2025), *Effective context engineering for AI agents* (above).
* McVeety and Hormati (2026), *Introducing the Open Knowledge Format*, Google Cloud Blog:
  the basis of Foveate's `okf` bundles.
* Coyle (2025), *Context Engineering: The New AI Frontier*, Medium, and Szapar (2026),
  *Context Engineering Research: 2026 Papers and Benchmarks*, iwoszapar.com: secondary
  overviews. We do not use them for numeric claims.
