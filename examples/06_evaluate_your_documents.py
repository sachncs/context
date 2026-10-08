"""Build a starter evaluation set from your own documents (no model needed).

`starter_items` makes answerable questions with their evidence pages,
unanswerable ones (a question paired with a document that lacks the answer)
and needle tests at five depths. Run them with `foveate.bench.longdoc.run`.
"""

from foveate import Document
from foveate.bench.longdoc import Example, starter_items


def main() -> None:
    policy_2024 = Document.load(
        b"Parental leave is 16 weeks.\fRemote work needs approval.",
        format="text",
        doc_id="policy_2024",
    )
    policy_2025 = Document.load(
        b"Employees are granted 26 weeks of paid parental leave.\f"
        b"Remote work is allowed.",
        format="text",
        doc_id="policy_2025",
    )
    examples = [
        Example(
            doc_id="policy_2025",
            question="How many weeks of parental leave are granted?",
            expected="26 weeks",
            evidence="Employees are granted 26 weeks of paid parental leave.",
        )
    ]
    items = starter_items([policy_2024, policy_2025], examples)
    for kind in ("answerable", "unanswerable", "needle"):
        count = sum(item.kind == kind for item in items)
        print(f"{kind:12s} {count}")
    first = items[0]
    print(first.question, "->", first.expected, "pages", first.gold_pages)


if __name__ == "__main__":
    main()
