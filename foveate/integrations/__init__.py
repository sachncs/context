"""Adapters that put foveate compression inside agent frameworks.

Import the module for your framework; each needs its framework installed:

* `foveate.integrations.pydantic_ai`  - history processor
* `foveate.integrations.adk`          - `before_model_callback`
* `foveate.integrations.langgraph`    - graph node / `pre_model_hook`

`foveate.integrations.history` holds the framework-neutral engine.
"""

from foveate.integrations.history import HistoryAdapter, HistoryCompressor

__all__ = ["HistoryAdapter", "HistoryCompressor"]
