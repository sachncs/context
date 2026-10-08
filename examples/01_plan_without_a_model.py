"""Plan a question about a 200-page report with no model and no cost.

`Foveator.plan` shows which pages would be sent in full, condensed or as an
outline, and how many tokens that is, before anything is paid for.
"""

import common

from foveate import Foveator, Runtime


def main() -> None:
    report = common.annual_report()
    print(f"{len(report.pages)} pages, {report.token_count:,} tokens")

    foveator = Foveator(Runtime.without_llm(), budget=4000)
    plan = foveator.plan(common.QUESTION, [report])
    print(
        f"send {plan.prompt_tokens:,} tokens instead of {plan.document_tokens:,}"
    )
    for page in plan.foveation.shown():
        print(f"  page {page.page:3d}  {page.tier.name}")

    # Choose pages yourself when you already know where to look.
    print(report.select("38-42").text(markers=True)[:80].replace("\n", " "))
    print(report.around(40, radius=1).numbers)


if __name__ == "__main__":
    main()
