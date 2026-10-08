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

## Compression research and how Foveate relates to it

Most compression research changes the model: it trains a compressor, reads hidden states, or
edits the key-value cache. That works when you host the model. Foveate works on text, so it
works with any API model, and its output (the pages it sent and the quotes it checked) can be
read and audited. The two approaches are complementary; the papers below also shaped how
Foveate is evaluated.

* **Learned latent compressors.** Wang et al. (2024), *In-Context Former*,
  [arXiv:2406.13618](https://arxiv.org/abs/2406.13618): cross-attention digest tokens give linear
  compression time; evaluated with BLEU-4, ROUGE and compression time on the PwC dataset.
  Liu et al., *Autoencoding-Free Context Compression via Contextual Semantic Anchors* (SAC),
  [arXiv:2510.08907](https://arxiv.org/abs/2510.08907); Ye et al., *ComprExIT*,
  [arXiv:2602.03784](https://arxiv.org/abs/2602.03784); and Li et al. (2026), *End-to-End Context
  Compression at Scale*, [arXiv:2606.09659](https://arxiv.org/abs/2606.09659). SAC and
  ComprExIT are evaluated on MRQA in and out of domain with exact match and F1 at 4x to 51x;
  ComprExIT notes that compression slots should cover complementary regions rather than the same
  one. The last paper reports time to first token and memory next to accuracy, and finds that an
  agent that can choose which compressed chunk to expand improves needle tasks.
  **Foveate takes** the evaluation protocol (exact match, F1, ratios of 4x and 8x, latency
  reported with accuracy) and the idea of selective expansion, which is the outline plus
  `need_pages` round in `Foveator`. These methods need weights or hidden states, so they are
  not options for API models.
* **Query-conditioned compression.** Ma et al. (2026), *Thinking as Compression*,
  [arXiv:2605.28713](https://arxiv.org/abs/2605.28713): a model writes a compact,
  question-specific trace and a second model answers from it; evaluated on NaturalQuestions,
  2WikiMQA, HotpotQA and MuSiQue with exact match and F1 at 4x and 8x. **Foveate takes** the
  protocol (it runs HotpotQA at 4x and 8x) and the principle that compression should depend on
  the question (`query` compressor, `query_aware` condensing).
* **Information-based pruning.** Li et al. (2023), *Compressing Context to Enhance Inference
  Efficiency of Large Language Models* (Selective Context),
  [arXiv:2310.06201](https://arxiv.org/abs/2310.06201): drops low-information content by
  self-information, found phrases a better unit than tokens or sentences, reported 36% less GPU
  memory and 32% lower latency at 50% compression with a BERTScore-F1 drop of 0.023, and used the
  full-context answer as the reference. **Foveate takes** the `selective` compressor (phrase-level,
  percentile-free budget fill, no model) and the "agreement with the full-context answer" metric.
* **Agent context policies.** Satish et al. (2026), *Beyond Token Savings: A Systematic Study of
  Context Compression in LLM Agents*, [arXiv:2609.32961](https://arxiv.org/abs/2609.32961): about
  35,000 runs on SWE-bench Verified and Terminal-Bench. Policies split into primitive (truncate,
  summarise, clear tool results), trigger and depth. Policies that used a third of the tokens
  took 20-80% longer on Terminal-Bench because of extra calls; billed cost depends on prefix
  caching (0.71-0.95x of full-context cost at 0.56-0.63x of its tokens); policies with similar
  success solved different tasks (one gained 7 and lost 8); and rankings flipped between
  models. **Foveate takes** the decomposition (`window`, `clear_tool_results` and `ppa` as
  primitives, `trigger` and `target` on `HistoryCompressor`), cost-aware evaluation (compressor
  tokens, latency, cached-token price) and the per-sample gained and lost counts.
* Dixit et al. (2026), *FOCUS: Training-Free Decision-Preserving Context Compression for LLM
  Agents*, [arXiv:2609.37590](https://arxiv.org/abs/2609.37590): scores spans by whether later
  decisions depend on them and adds a defensive pass that rescues spans carrying negative
  constraints or state; reports +8.9 points on AppWorld with 35-48% lower peak tokens.
  **Foveate takes** the defensive rule: `clear_tool_results` keeps results that look like
  failures and honours `exclude`.
* **Verifiable compression.** Trukhina and Vashkelis (2026), *Compress the Context, Keep the
  Commitments*, [arXiv:2605.17304](https://arxiv.org/abs/2605.17304): represent state as typed,
  source-grounded atoms; measure Critical Atom Recall, Weighted Atom Recall and commitment
  density; distinguish omission, weakening and mutation; keep raw spans for low-confidence
  facts. Free prose compressed 68% but lost 39% of commitments. **Foveate takes** the metrics
  (`foveate.bench.scoring`) and a failure taxonomy (kept, mutated, omitted), so compression is
  measured by what survives, not only by whether one question is still answered.

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
