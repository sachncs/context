"""Shared helpers for the examples.

Examples that call a model read their configuration from the environment
(see the README). For a quick try with an OpenAI-compatible endpoint:

    export FOVEATE_BACKEND=openai
    export FOVEATE_BASE_URL=https://integrate.api.nvidia.com/v1
    export FOVEATE_MODEL=openai/gpt-oss-20b
    export FOVEATE_OPTIONS='{"reasoning_effort": "low"}'
    export OPENAI_API_KEY=<your key>
"""

from foveate import Runtime


def long_document(facts: int = 6) -> str:
    """Builds a long document whose key facts are buried in filler."""
    filler = (
        "The committee reviewed logistics, parking and catering without "
        "reaching any decision. "
    ) * 12
    parts = []
    for number in range(facts):
        parts.append(filler)
        parts.append(
            f"Fact {number}: the reference value is {number * 7 + 100}. "
        )
    return "".join(parts)


def model_runtime() -> Runtime:
    """Returns a runtime configured from `FOVEATE_*` environment variables."""
    return Runtime.from_env()
