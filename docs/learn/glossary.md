---
title: "Glossary"
description: "Plain definitions of the terms used across the Foveate docs."
---

**Abstain.** To answer "not found" instead of guessing. Foveate reports it as `Answer.abstained`.

**Atom.** A single fact that must survive compression (a number, a name, a decision). Used to measure
how much of what mattered is still there after shrinking.

**Budget.** The most tokens you allow for some part of the prompt (for example the evidence block).

**Cache (prompt cache).** A provider feature that charges less for the start of a prompt it has seen
recently. Matches from the beginning of the prompt only.

**Cache (response cache).** Foveate's local store of past model replies, keyed by the exact request.

**Chunk.** A slice of a page (up to 256 tokens) that remembers its page and heading. The unit of search.

**Citation.** A (document, page, quote) triple the model gives to show where an answer came from.

**Condensed.** The tier where a page is replaced by its most useful sentences.

**Context engineering.** Deciding what goes into a model's prompt: selecting, shaping, compressing,
remembering, verifying and measuring.

**Context rot.** The drop in a model's ability to use information as the prompt grows.

**Context window.** The most tokens a model can read in one request.

**Embedding.** A list of numbers that represents the meaning of a text, so similar meanings are close.

**Foveation.** Giving each page the detail it deserves: full, condensed, outline or none.

**Grounded.** Having at least one citation, every one of which was found on its cited page.

**Haystack and needle.** A long text (haystack) with one fact (needle) hidden in it, used to test whether
a model can find it.

**Hallucination.** A fluent statement that is not supported by the source or by fact.

**Outline.** The tier where a page is reduced to one line: its number and heading.

**Page score.** How relevant a page is to a question, from the retriever.

**Plan.** The allocation and cost estimate Foveate computes before calling a model.

**Prompt injection.** Text in a document that tries to give the model instructions.

**Query expansion.** Having a model write alternative search phrasings of a question.

**RAG (retrieval-augmented generation).** Retrieving relevant text and giving it to a model to answer from.

**Runtime.** The object that owns the backend, cache, tokenizer and settings for all model calls.

**Token.** A piece of text a model reads or writes; about four characters of English.

**Tool result.** What an agent's tool returns, added to the conversation.
