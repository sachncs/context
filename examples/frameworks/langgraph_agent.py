"""LangGraph: page tools and a history budget as a pre-model hook.

pip install "git+https://github.com/sachncs/foveate" "git+https://github.com/sachncs/foveate#subdirectory=integrations/langgraph" langchain-openai
"""

import os
import sys
from pathlib import Path

import foveate_langgraph as foveate_lg
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


def main() -> None:
    report = common.annual_report()
    with common.model_runtime() as runtime:
        model = ChatOpenAI(
            model=runtime.model,
            base_url=os.environ.get("FOVEATE_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        agent = create_react_agent(
            model,
            foveate_lg.document_tools([report]),
            pre_model_hook=foveate_lg.compression_node(runtime, budget=6000),
        )
        result = agent.invoke({"messages": [("user", common.QUESTION)]})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
