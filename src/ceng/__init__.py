"""ceng: context engineering for LLMs.

Typical use:

    ctx = Context.from_dicts(messages, Runtime.from_env())
    smaller = ctx.compress("ppa", budget=4000)
"""

from ceng.context import Context
from ceng.errors import CengError
from ceng.messages import Message, Role
from ceng.runtime import Runtime

__version__ = "2.0.0.dev0"

__all__ = ["CengError", "Context", "Message", "Role", "Runtime", "__version__"]
