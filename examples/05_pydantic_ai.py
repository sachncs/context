"""Keep a Pydantic AI agent's history within a token budget.

Requires: pip install "foveate[openai]" pydantic-ai
Configure the model via the FOVEATE_* variables in common.py plus OPENAI_API_KEY.
"""

import asyncio
import os

import common
from pydantic_ai import Agent
from pydantic_ai import messages as pai
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from foveate.integrations import pydantic_ai as foveate_pai

FILLER = (
    "We reviewed logistics, parking and catering without any decision. " * 20
)
TURNS = [
    ("user", f"Remember: the team locker number is 7391-KAPPA. {FILLER}"),
    ("assistant", f"Noted. {FILLER}"),
    ("user", f"Next topic. {FILLER}"),
    ("assistant", f"Understood. {FILLER}"),
]


async def main() -> None:
    with common.model_runtime() as runtime:
        model = OpenAIChatModel(
            runtime.model,
            provider=OpenAIProvider(
                base_url=os.environ.get("FOVEATE_BASE_URL"),
                api_key=os.environ["OPENAI_API_KEY"],
            ),
        )
        processor = foveate_pai.history_processor(
            runtime, budget=300, keep_last=2
        )
        agent = Agent(
            model,
            capabilities=[ProcessHistory(processor)],
            model_settings={
                "openai_reasoning_effort": "low",
                "max_tokens": 2000,
            },
        )
        history = [
            pai.ModelRequest(parts=[pai.UserPromptPart(content=text)])
            if kind == "user"
            else pai.ModelResponse(parts=[pai.TextPart(content=text)])
            for kind, text in TURNS
        ]
        result = await agent.run(
            "What is the team locker number?", message_history=history
        )
        report = processor.compressor.reports[-1]
        print(
            f"history {report.original_tokens} -> {report.final_tokens} tokens"
        )
        print("answer:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
