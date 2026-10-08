"""Framework-neutral building blocks for agents.

* `HistoryCompressor` keeps a message history inside a token budget; a
  `HistoryAdapter` teaches it one framework's message type.
* `DocumentTools` gives an agent `read_pages`, `search_document` and
  `document_outline` as plain typed functions.

Foveate itself depends on no agent framework. Small adapter packages in the
`integrations/` directory of the repository wire these pieces into Pydantic AI,
Google ADK, LangGraph and Strands Agents.
"""

from foveate.agents.history import HistoryAdapter, HistoryCompressor
from foveate.agents.tools import DocumentTools

__all__ = ["DocumentTools", "HistoryAdapter", "HistoryCompressor"]
