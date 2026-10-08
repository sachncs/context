"""Google ADK: page tools and a history budget before each model call.

pip install foveate foveate-adk
export GOOGLE_API_KEY=...
"""

import asyncio
import sys
from pathlib import Path

import foveate_adk
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


async def main() -> None:
    report = common.annual_report()
    with common.model_runtime() as runtime:
        agent = LlmAgent(
            name="analyst",
            model="gemini-2.0-flash",
            instruction="Answer from the document tools. Cite pages as [doc p.N].",
            tools=foveate_adk.document_tools([report]),
            before_model_callback=foveate_adk.model_callback(runtime, 6000),
        )
        runner = InMemoryRunner(agent=agent, app_name="foveate-demo")
        session = await runner.session_service.create_session(
            app_name="foveate-demo", user_id="u"
        )
        message = types.Content(
            role="user", parts=[types.Part(text=common.QUESTION)]
        )
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=message
        ):
            if event.is_final_response() and event.content:
                print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
