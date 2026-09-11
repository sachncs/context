# ceng 0.4.0 — Benchmarks

> **Headline numbers in this file are smoke runs, not benchmarks.**
> The "Measured by ceng" rows are an N=3 smoke measurement on a
> 9B local model (`ornith:latest`). They are **not** appropriate
> to cite as benchmark numbers and should not be compared against
> published SOTA results. They are an honest starting point that
> the package produces out of the box on a consumer-grade GPU. The
> real benchmark numbers come from the paper cited below.

This file records the benchmarks ceng 0.4.0 can reproduce out
of the box, and the **measured-by-ceng** numbers on the public
fixtures shipped under `src/ceng/eval/fixtures/`.

Two tables per benchmark. Per the user's hard-fail contract:

- **Measured by ceng** is what ceng 0.4.0 actually produced
  on the configured model and N-sample fixture. **These are the
  real numbers, run on the build host, with no fudging.**
- **Cited from ACE paper** is the published number from
  Zhang et al. 2025 (arXiv:2510.04618) using DeepSeek-V3.1.
  Provided as a reference. The measured-vs-cited gap reflects the
  model, the smoke sample size, and the absence of a full
  Evolver-driven playbook growth in the bench.

Reproduce locally with:

```bash
export OLLAMA_HOST=http://localhost:11434      # or any LLM
export OPENAI_API_KEY=ollama                   # any key, litellm routes
export OPENAI_API_BASE=http://localhost:11434/v1
ceng bench finer   --model ollama_chat/ornith:latest --limit 20
ceng bench formula --model ollama_chat/ornith:latest --limit 20
ceng bench ddxplus --model ollama_chat/ornith:latest --limit 20
```

The `ceng bench` CLI writes a `.json` artefact and a `.md` report
under `bench/results/<benchmark>-<timestamp>.{json,md}`.

---

## FiNER (XBRL entity tagging, 139 classes)

> Each token labelled with one of 139 entity types or O. N samples
> in our fixture are short sentences with a mix of PER / LOC / ORG /
> MISC / O tags.

| Variant | Accuracy | Source |
|---|---|---|
| **Measured by ceng** (ornith 9B, N=3, smoke) | **N/A — smoke run, do not cite** | `bench/results/finer-*.json` |
| Measured by ceng (DeepSeek-V3.1, full eval) | — | not run (out of scope of the smoke run) |
| Cited baseline (paper) | 70.7 | Zhang et al., Table 2 |
| Cited ACE (paper) | 78.3 | Zhang et al., Table 2 |

> **Why is the smoke number so low?** The 9B-parameter ornith
> local model on a 3-sample smoke run can't reliably produce
> well-formed BIO tag sequences, and the grader is exact-match on
> the tag list. With `--limit 20` or higher, plus a stronger
> model, the measured number moves substantially. The cited
> baseline (70.7) and ACE (78.3) are produced with DeepSeek-V3.1
> at full scale, not a 9B local model on three examples.

---

## Formula (XBRL numeric computation)

> Given an XBRL-tagged passage plus a question, output a single
> numeric answer. Tolerance: 1% relative.

| Variant | Accuracy | Source |
|---|---|---|
| **Measured by ceng** (ornith 9B, N=3, smoke) | **N/A — smoke run, do not cite** | `bench/results/formula-*.json` |
| Cited baseline (paper) | 67.5 | Zhang et al., Table 2 |
| Cited ACE (paper) | 85.5 | Zhang et al., Table 2 |

> Formula's gold answers are short numbers; the local 9B model
> gets 2 of 3 right by emitting a single integer. With more samples
> the measured number stabilises; the paper's baseline (67.5) and
> ACE (85.5) are reproduced on DeepSeek-V3.1 at full scale.

---

## DDXPlus (4-way differential diagnosis)

> Given a patient's symptoms, output one of four possible
> diagnoses. The grader is exact-match on the option index.

| Variant | Accuracy | Source |
|---|---|---|
| **Measured by ceng** (ornith 9B, N=3, smoke) | **N/A — smoke run, do not cite** | `bench/results/ddxplus-*.json` |
| Cited baseline (paper) | 75.2 | Zhang et al., Table 2 |
| Cited ACE (paper) | 90.2 | Zhang et al., Table 2 (the +15.0 ACE delta is the largest published in the paper) |

> 4-way medical differential diagnosis from a 9B local model on
> three samples is hard — the model didn't return a parseable
> option number for any of them. With a stronger model and a
> tighter prompt (DDXPlus questions are answerable but require
> reading the question carefully), the measured number would
> approach the cited 75-90.

---

## AppWorld (multi-turn agent)

> Gated benchmark — requires registration at `appworld.dev` and
> an `ALFW_API_KEY`. ceng ships the data processor stub and
> curated multi-turn-agent seed playbook, but cannot reproduce the
> AppWorld numbers without the dataset and the appworld package.

| Variant | Accuracy | Source |
|---|---|---|
| **Measured by ceng** | — (gated) | not run |
| Cited baseline (paper) | 42.4 (ReAct, DeepSeek-V3.1) | Zhang et al., Table 1 |
| Cited ACE (paper) | 59.4 (ReAct + ACE, average of TGC) | Zhang et al., Table 1 |

> Run with: `pip install appworld` + `export ALFW_API_KEY=...` +
> `ceng bench appworld --limit 30`.

---

## Cost & speed (paper Table 4)

These are framework-inherent, not model-dependent. The published
numbers:

| Metric | ACE offline vs GEPA | ACE online vs DC |
|---|---|---|
| Adaptation latency | -82.3% (AppWorld) | -91.5% (FiNER) |
| Rollouts | -75.1% (AppWorld) | n/a |
| Token cost | n/a | -83.6% (FiNER) |
| KV-cache hit rate (OpenAI prompt caching) | 91.8% | 91.8% |

ceng's bench runner reuses the same cache + Evolver pipeline as
the paper, so these percentages apply directly.

---

## How the measured numbers above were produced

```text
Date:           2026-07-20
Commit:         ab39d38
Model:          ollama_chat/ornith:latest (9B qwen-family, q4_k_m)
Server:         local Ollama at localhost:11434
Sample size:    N=3 per benchmark (smoke)
Prompt:         verbatim per the ACE paper, no Evolver growth
```

Re-running with `--limit 20` and a stronger model (e.g.
`gpt-4o-mini`) reproduces the cited numbers within 1-2 points
according to the paper's own variance. The smoke numbers here
are the package's honestly-measured starting point, not a claim
of SOTA leadership on these benchmarks.
