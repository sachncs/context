"""ceng: context engineering for LLMs.

One noun, `Context`, and strategies behind it:

    from ceng import Context, Runtime

    context = Context.from_dicts(messages, Runtime.from_env())
    smaller = context.compress("ppa", budget=4000)
    print(smaller.report)

Subpackages: `compression` (strategies), `verification`, `okf` (Open
Knowledge Format), `stores` (notes), `evolution` (ACE playbooks), `bench`,
`backends`, `cache`, `tokenizers`.
"""

from ceng import errors
from ceng.compression import Budget, CompressionReport, Compressor, Overflow
from ceng.context import Context
from ceng.messages import Message, Role
from ceng.runtime import Runtime

__version__ = "2.0.0.dev0"

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
