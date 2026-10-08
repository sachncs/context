"""Helpers shared by the examples.

Examples that call a model read their settings from the environment, for any
OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, NVIDIA, Together, ...):

    export FOVEATE_BASE_URL=https://api.openai.com/v1
    export FOVEATE_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=...

Without FOVEATE_MODEL the examples that need a model still run their offline
part and say what to set.
"""

import os

from foveate import Document, Runtime

QUESTION = "What were capital expenditures in fiscal 2018?"


def has_model() -> bool:
    """Returns whether a model is configured through the environment."""
    return bool(os.environ.get("FOVEATE_MODEL"))


def model_runtime() -> Runtime:
    """Returns a runtime built from the `FOVEATE_*` variables."""
    return Runtime.from_env()


def annual_report(pages: int = 200) -> Document:
    """Builds a long report with one relevant sentence on page 40."""
    texts = [
        f"Section {n}. " + "Routine operations are described here. " * 60
        for n in range(1, pages + 1)
    ]
    texts[39] += (
        " Capital expenditures were 1,577 million dollars in fiscal 2018."
    )
    return Document.load(
        "\f".join(texts).encode(), format="text", doc_id="annual_report"
    )
