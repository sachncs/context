"""Ask a question and get an answer with verified page citations.

Needs a model (see common.py). Without one it prints the plan only.
"""

import common

from foveate import Foveator


def main() -> None:
    report = common.annual_report()
    if not common.has_model():
        print("Set FOVEATE_MODEL (see common.py) to ask the model.")
        return
    with common.model_runtime() as runtime:
        foveator = Foveator(runtime, budget=4000)
        answer = foveator.ask(common.QUESTION, [report])
    print(answer.text)
    for citation in answer.citations:
        print(
            f"  {citation.doc_id} p.{citation.page} "
            f"verified={citation.verified}: {citation.quote!r}"
        )
    print("grounded:", answer.grounded, "| abstained:", answer.abstained)
    print("tokens:", answer.usage.total_tokens, "| rounds:", answer.rounds)


if __name__ == "__main__":
    main()
