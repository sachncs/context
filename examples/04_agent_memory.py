"""Give an agent working notes and durable memory, with recall by query.

Session notes are scratch space for one task; facts survive across sessions.
`consolidate` (one model call) turns notes into facts; recall needs no model.
"""

from foveate import Memory
from foveate.stores import MemoryNotesStore


def main() -> None:
    memory = Memory(MemoryNotesStore())
    memory.remember("Deployments run in eu-west-1.")
    memory.remember("The user prefers short answers.")
    memory.remember("Checked the billing export", session="s1")

    for note in memory.recall("where do deployments run", k=2):
        print("recalled:", note.content)

    message = memory.message("deployments", budget=80)
    print(message.content if message else "nothing relevant")


if __name__ == "__main__":
    main()
