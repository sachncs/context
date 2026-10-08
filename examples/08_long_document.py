"""Ask a question about a long document: plan first, then answer with citations.

Without a model configured this prints the plan only (no calls are made). Set
the FOVEATE_* variables from common.py to also get a grounded answer.
"""

import os

import common

from foveate import Document, Foveator, Runtime

QUESTION = "What were capital expenditures in fiscal 2018?"


def build_document() -> Document:
    """Builds a 200-page report with one relevant sentence on page 40."""
    pages = [
        f"Section {n}. " + "Routine operations are described here. " * 60
        for n in range(1, 201)
    ]
    pages[39] += (
        " Capital expenditures were 1,577 million dollars in fiscal 2018."
    )
    return Document.load(
        "\f".join(pages).encode(), format="text", doc_id="annual_report"
    )


def main() -> None:
    document = build_document()
    print(f"{len(document.pages)} pages, {document.token_count:,} tokens")

    plan = Foveator(Runtime.without_llm(), budget=4000).plan(
        QUESTION, [document]
    )
    print(
        f"plan: send {plan.prompt_tokens:,} tokens instead of "
        f"{plan.document_tokens:,}"
    )
    for page in plan.foveation.shown():
        print(f"  page {page.page:3d}  {page.tier.name}")

    if not os.environ.get("FOVEATE_MODEL"):
        print("set FOVEATE_MODEL (see common.py) to ask the model")
        return
    with common.model_runtime() as runtime:
        answer = Foveator(runtime, budget=4000).ask(QUESTION, [document])
    print(answer.text)
    print("cited pages:", answer.pages, "grounded:", answer.grounded)


if __name__ == "__main__":
    main()
