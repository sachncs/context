"""Keep a LangGraph agent's history within a token budget.

Requires: pip install "ceng-context[openai]" langgraph langchain-openai
Configure the model via the CENG_* variables in common.py plus OPENAI_API_KEY.
"""

import asyncio
import os

import common
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ceng.integrations import langgraph as ceng_lg

FILLER = (
    "We reviewed logistics, parking and catering without any decision. " * 20
)
TURNS = [
    HumanMessage(
        content=f"Remember: the team locker number is 7391-KAPPA. {FILLER}"
    ),
    AIMessage(content=f"Noted. {FILLER}"),
    HumanMessage(content=f"Next topic. {FILLER}"),
    AIMessage(content=f"Understood. {FILLER}"),
]


async def main() -> None:
    with common.model_runtime() as runtime:
        node = ceng_lg.compression_node(runtime, budget=300, keep_last=2)
        model = ChatOpenAI(
            model=runtime.model,
            base_url=os.environ.get("CENG_BASE_URL"),
            api_key=os.environ["OPENAI_API_KEY"],
            extra_body={"reasoning_effort": "low"},
            max_tokens=2000,
        )
        # As a pre_model_hook the model sees the compressed history while the
        # stored state keeps the full transcript. The node also works in any
        # StateGraph (use persist=True to replace the stored history).
        agent = create_react_agent(model, [], pre_model_hook=node)
        result = await agent.ainvoke(
            {
                "messages": [
                    *TURNS,
                    HumanMessage(content="What is the team locker number?"),
                ]
            }
        )
        report = node.compressor.reports[-1]
        print(
            f"history {report.original_tokens} -> {report.final_tokens} tokens"
        )
        print("answer:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
