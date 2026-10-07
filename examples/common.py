"""Shared helpers for the examples.

Examples that call a model read their configuration from the environment
(see the README). For a quick try with an OpenAI-compatible endpoint:

    export CENG_BACKEND=openai
    export CENG_BASE_URL=https://integrate.api.nvidia.com/v1
    export CENG_MODEL=openai/gpt-oss-20b
    export CENG_OPTIONS='{"reasoning_effort": "low"}'
    export OPENAI_API_KEY=<your key>
"""

from ceng import Runtime


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
    """Returns a runtime configured from `CENG_*` environment variables."""
    return Runtime.from_env()
