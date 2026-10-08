"""Strands Agents: page tools and a conversation manager that compresses.

pip install foveate foveate-strands "strands-agents[openai]"
"""

import os
import sys
from pathlib import Path

import foveate_strands
from strands import Agent
from strands.models.openai import OpenAIModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


def main() -> None:
    report = common.annual_report()
    with common.model_runtime() as runtime:
        model = OpenAIModel(
            client_args={
                "base_url": os.environ.get("FOVEATE_BASE_URL"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
            model_id=runtime.model,
        )
        agent = Agent(
            model=model,
            tools=foveate_strands.document_tools([report]),
            conversation_manager=foveate_strands.conversation_manager(
                runtime, budget=6000
            ),
        )
        print(agent(common.QUESTION))


if __name__ == "__main__":
    main()
