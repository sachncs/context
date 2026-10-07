"""Adapters that put ceng compression inside agent frameworks.

Import the module for your framework; each needs its framework installed:

* `ceng.integrations.pydantic_ai`  - history processor
* `ceng.integrations.adk`          - `before_model_callback`
* `ceng.integrations.langgraph`    - graph node / `pre_model_hook`

`ceng.integrations.history` holds the framework-neutral engine.
"""

from ceng.integrations.history import HistoryAdapter, HistoryCompressor

__all__ = ["HistoryAdapter", "HistoryCompressor"]
