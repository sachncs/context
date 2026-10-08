"""foveate: context engineering for LLM applications.

Decide what goes into the model's context window: load documents by page,
select the relevant pages, keep detail only where it matters, answer with
verified citations, and measure the result.

    import foveate

    runtime = foveate.Runtime.from_env()
    report = foveate.Document.load("annual-report.pdf")      # 200 pages
    answer = foveate.Foveator(runtime).ask(
        "What was capital expenditure in 2018?", [report]
    )
    answer.text, answer.citations, answer.grounded

Chat histories and arbitrary messages go through `Context`:

    smaller = foveate.Context.from_dicts(messages, runtime).compress(
        "ushape+ppa", budget=4000
    )

Subpackages: `documents`, `selection`, `grounding`, `compression`,
`verification`, `integrations` (Pydantic AI, Google ADK, LangGraph), `bench`,
`okf`, `stores`, `evolution`, `backends`, `cache`, `tokenizers`.
"""

from foveate import errors
from foveate.compression import Budget, CompressionReport, Compressor, Overflow
from foveate.context import Context
from foveate.documents import Document
from foveate.foveator import Foveator, Index, Plan
from foveate.grounding import Answer, Citation
from foveate.memory import Memory
from foveate.messages import Message, Role
from foveate.runtime import Runtime

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Budget",
    "Citation",
    "CompressionReport",
    "Compressor",
    "Context",
    "Document",
    "Foveator",
    "Index",
    "Memory",
    "Message",
    "Overflow",
    "Plan",
    "Role",
    "Runtime",
    "__version__",
    "errors",
]
