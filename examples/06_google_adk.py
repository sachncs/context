"""Keep a Google ADK agent's request contents within a token budget.

Requires: pip install "ceng-context[openai]" google-adk litellm
Configure the model via the CENG_* variables in common.py plus OPENAI_API_KEY.
"""

import asyncio
import os

import common
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ceng.integrations import adk as ceng_adk

FILLER = (
    "We reviewed logistics, parking and catering without any decision. " * 20
)
TURNS = [
    ("user", f"Remember: the team locker number is 7391-KAPPA. {FILLER}"),
    ("model", f"Noted. {FILLER}"),
    ("user", f"Next topic. {FILLER}"),
    ("model", f"Understood. {FILLER}"),
]


async def main() -> None:
    with common.model_runtime() as runtime:
        callback = ceng_adk.model_callback(runtime, budget=300, keep_last=2)
        agent = LlmAgent(
            name="assistant",
            model=LiteLlm(
                model=f"openai/{runtime.model}",
                api_base=os.environ.get("CENG_BASE_URL"),
                api_key=os.environ["OPENAI_API_KEY"],
                extra_body={"reasoning_effort": "low"},
                max_tokens=2000,
            ),
            instruction="Answer briefly using the conversation.",
            before_model_callback=callback,
        )
        sessions = InMemorySessionService()
        session = await sessions.create_session(
            app_name="demo", user_id="u", session_id="s"
        )
        for role, text in TURNS:
            await sessions.append_event(
                session,
                Event(
                    author="u" if role == "user" else "assistant",
                    content=types.Content(
                        role=role, parts=[types.Part(text=text)]
                    ),
                ),
            )
        runner = Runner(agent=agent, app_name="demo", session_service=sessions)
        async for event in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="What is the team locker number?")],
            ),
        ):
            if event.is_final_response() and event.content:
                print("answer:", event.content.parts[0].text)
        report = callback.compressor.reports[-1]
        print(
            f"contents {report.original_tokens} -> {report.final_tokens} tokens"
        )


if __name__ == "__main__":
    asyncio.run(main())
