"""foveate: context engineering for LLMs.

One noun, `Context`, and strategies behind it:

    from foveate import Context, Runtime

    context = Context.from_dicts(messages, Runtime.from_env())
    smaller = context.compress("ppa", budget=4000)
    print(smaller.report)

Subpackages: `compression` (strategies), `verification`, `okf` (Open
Knowledge Format), `stores` (notes), `evolution` (ACE playbooks), `bench`,
`backends`, `cache`, `tokenizers`.
"""

from foveate import errors
from foveate.compression import Budget, CompressionReport, Compressor, Overflow
from foveate.context import Context
from foveate.messages import Message, Role
from foveate.runtime import Runtime

__version__ = "0.1.0"

__all__ = [
    "Budget",
    "CompressionReport",
    "Compressor",
    "Context",
    "Message",
    "Overflow",
    "Role",
    "Runtime",
    "__version__",
    "errors",
]
