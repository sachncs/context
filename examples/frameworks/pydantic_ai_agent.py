"""Pydantic AI: page tools and a history budget.

pip install foveate foveate-pydantic-ai "pydantic-ai-slim[openai]"
"""

import os
import sys
from pathlib import Path

import foveate_pydantic_ai as foveate_pai
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


def main() -> None:
    report = common.annual_report()
    with common.model_runtime() as runtime:
        model = OpenAIChatModel(
            runtime.model,
            provider=OpenAIProvider(
                base_url=os.environ.get("FOVEATE_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY"),
            ),
        )
        agent = Agent(
            model,
            tools=foveate_pai.document_tools([report]),
            capabilities=[
                ProcessHistory(foveate_pai.history_processor(runtime, 6000))
            ],
        )
        print(agent.run_sync(common.QUESTION).output)


if __name__ == "__main__":
    main()
