"""Save/load a context as an OKF bundle; offload old turns to a notes store."""

import tempfile
from pathlib import Path

from ceng import Context, Message, Role, Runtime
from ceng.backends import ScriptedBackend
from ceng.cache import MemoryCache
from ceng.stores import FilesystemNotesStore


def main() -> None:
    runtime = Runtime(backend=ScriptedBackend(), cache=MemoryCache())
    turns = tuple(
        Message(Role.USER, f"message {i}: " + "detail " * 60) for i in range(10)
    )
    context = Context(turns, runtime)

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        store = FilesystemNotesStore(root / "notes")

        # Nothing is lost: the middle goes to a note, a pointer stays behind.
        slim = context.compress(
            "offload", budget=300, store=store, head=1, tail=2
        )
        pointer = slim.messages[1].content
        print(pointer)
        print("recoverable chars:", len(store.entries()[0].content))

        slim.save(root / "context-bundle")  # OKF markdown bundle
        again = Context.load(root / "context-bundle", runtime=runtime)
        assert again.messages == slim.messages and again.report == slim.report
        print(
            "bundle files:",
            sorted(p.name for p in (root / "context-bundle").rglob("*.md"))[:3],
        )


if __name__ == "__main__":
    main()
