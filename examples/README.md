# Examples

Run any of them from this directory. The first three need no model.

| File | What it shows | Needs a model |
|---|---|---|
| `01_plan_without_a_model.py` | Plan a question about a 200-page report; choose pages yourself | no |
| `02_ask_with_citations.py` | Ask and get verified page citations | yes |
| `03_compress_without_a_model.py` | Shrink a chat and a tool result with model-free methods | no |
| `04_agent_memory.py` | Session notes, durable facts and recall | no |
| `05_resilient_backend.py` | Retries, timeout and breaker around any backend | no |
| `06_evaluate_your_documents.py` | Build a starter evaluation set from your own files | no |
| `frameworks/` | The same ideas inside Pydantic AI, Google ADK, LangGraph and Strands | yes |

Models: any OpenAI-compatible endpoint.

```bash
export FOVEATE_BASE_URL=https://api.openai.com/v1   # or your vLLM / Ollama URL
export FOVEATE_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
```

The framework examples use the adapter packages in `integrations/`; Foveate itself
does not depend on any agent framework.
