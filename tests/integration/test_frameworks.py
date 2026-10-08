"""foveate inside real agent frameworks, talking to a real model.

Each test builds a conversation whose *old* turns hold a secret, runs a real
agent whose history hook compresses those turns, and checks that (a) the hook
really shrank the history and (b) the model can still answer from it.
Frameworks that are not installed are skipped.
"""

import asyncio

import pytest

from tests.integration import conftest

ACCESS_CODE = "7391-KAPPA"
FILLER = (
    "We went over the quarterly logistics, the parking rota, the catering "
    "contract, and the recurring meeting schedule without any decisions. "
)
OLD_TURNS = [
    (
        "user",
        f"Please remember this: the team locker number is {ACCESS_CODE}. {FILLER * 3}",
    ),
    (
        "assistant",
        f"Noted, I will remember the team locker number. {FILLER * 3}",
    ),
    ("user", f"Next topic. {FILLER * 4}"),
    ("assistant", f"Understood. {FILLER * 4}"),
    ("user", f"Another topic. {FILLER * 4}"),
    ("assistant", f"Understood again. {FILLER * 4}"),
]
QUESTION = "What is the team locker number? Reply with the code only."


@pytest.fixture(scope="module")
def runtime():
    from foveate import runtime as runtime_lib
    from foveate.cache import sqlite as cache_sqlite

    rt = runtime_lib.Runtime(
        backend=conftest.make_backend(),
        model=conftest.MODEL,
        cache=cache_sqlite.SqliteCache(conftest.CACHE_DIR),
        options={"reasoning_effort": "low"},
        concurrency=4,
    )
    yield rt
    rt.close()


def test_pydantic_ai_agent_with_history_processor(runtime):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai import messages as pai
    from pydantic_ai.capabilities import ProcessHistory
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from foveate.integrations import pydantic_ai as foveate_pai

    model = OpenAIChatModel(
        conftest.MODEL,
        provider=OpenAIProvider(
            base_url=conftest.BASE_URL, api_key=conftest.API_KEY
        ),
    )
    processor = foveate_pai.history_processor(runtime, budget=450, keep_last=2)
    agent = Agent(
        model,
        capabilities=[ProcessHistory(processor)],
        model_settings={"openai_reasoning_effort": "low", "max_tokens": 2000},
    )
    history = [
        pai.ModelRequest(parts=[pai.UserPromptPart(content=text)])
        if kind == "user"
        else pai.ModelResponse(parts=[pai.TextPart(content=text)])
        for kind, text in OLD_TURNS
    ]
    result = asyncio.run(agent.run(QUESTION, message_history=history))
    report = processor.compressor.reports[-1]
    assert report.final_tokens < report.original_tokens
    assert conftest.fold(ACCESS_CODE) in conftest.fold(result.output), (
        result.output
    )


def test_adk_agent_with_before_model_callback(runtime):
    pytest.importorskip("google.adk")
    pytest.importorskip("litellm")
    from google.adk.agents import LlmAgent
    from google.adk.events import Event
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from foveate.integrations import adk as foveate_adk

    callback = foveate_adk.model_callback(runtime, budget=450, keep_last=2)
    agent = LlmAgent(
        name="assistant",
        model=LiteLlm(
            model=f"openai/{conftest.MODEL}",
            api_base=conftest.BASE_URL,
            api_key=conftest.API_KEY,
            extra_body={"reasoning_effort": "low"},
            max_tokens=2000,
        ),
        instruction="Answer briefly using the conversation.",
        before_model_callback=callback,
    )

    async def go() -> str:
        sessions = InMemorySessionService()
        session = await sessions.create_session(
            app_name="foveate-test", user_id="u", session_id="s"
        )
        for kind, text in OLD_TURNS:
            content = types.Content(
                role="user" if kind == "user" else "model",
                parts=[types.Part(text=text)],
            )
            await sessions.append_event(
                session,
                Event(
                    author="u" if kind == "user" else "assistant",
                    content=content,
                ),
            )
        runner = Runner(
            agent=agent, app_name="foveate-test", session_service=sessions
        )
        answer = ""
        async for event in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(
                role="user", parts=[types.Part(text=QUESTION)]
            ),
        ):
            if (
                event.is_final_response()
                and event.content
                and event.content.parts
            ):
                answer = event.content.parts[0].text or ""
        return answer

    answer = asyncio.run(go())
    report = callback.compressor.reports[-1]
    assert report.final_tokens < report.original_tokens
    assert conftest.fold(ACCESS_CODE) in conftest.fold(answer), answer


def test_langgraph_react_agent_with_pre_model_hook(runtime):
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_openai")
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from foveate.integrations import langgraph as foveate_lg

    node = foveate_lg.compression_node(runtime, budget=450, keep_last=2)
    model = ChatOpenAI(
        model=conftest.MODEL,
        base_url=conftest.BASE_URL,
        api_key=conftest.API_KEY,
        extra_body={"reasoning_effort": "low"},
        max_tokens=2000,
    )
    agent = create_react_agent(model, [], pre_model_hook=node)
    messages = [
        (HumanMessage if kind == "user" else AIMessage)(content=text)
        for kind, text in OLD_TURNS
    ] + [HumanMessage(content=QUESTION)]
    result = asyncio.run(agent.ainvoke({"messages": messages}))
    report = node.compressor.reports[-1]
    assert report.final_tokens < report.original_tokens
    assert conftest.fold(ACCESS_CODE) in conftest.fold(
        result["messages"][-1].content
    )
    # model input was compressed but the stored transcript is complete
    assert len(result["messages"]) >= len(messages)


def test_langgraph_node_can_persist_the_compressed_history(runtime):
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    from foveate.integrations import langgraph as foveate_lg

    node = foveate_lg.compression_node(
        runtime, budget=450, keep_last=2, persist=True
    )
    builder = StateGraph(MessagesState)
    builder.add_node("compress", node)
    builder.add_edge(START, "compress")
    builder.add_edge("compress", END)
    graph = builder.compile()
    messages = [
        (HumanMessage if kind == "user" else AIMessage)(content=text)
        for kind, text in OLD_TURNS
    ] + [HumanMessage(content=QUESTION)]
    result = asyncio.run(graph.ainvoke({"messages": messages}))
    stored = sum(len(m.content) for m in result["messages"])
    assert stored < sum(len(m.content) for m in messages)
    assert result["messages"][-1].content == QUESTION
    assert conftest.fold(ACCESS_CODE) in conftest.fold(
        " ".join(m.content for m in result["messages"])
    )
