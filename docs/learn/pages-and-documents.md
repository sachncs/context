---
title: "Pages and documents"
description: "How Foveate represents a document as numbered pages, how to load one, and how to pick pages yourself."
---

Everything in Foveate is addressed by **document id and page number**. That one decision makes
retrieval, citations, tools and evaluation line up: when the model says "page 60", you can open
page 60.

## What a document is

A `Document` is an immutable list of pages. Each `Page` has a number, its text, its token count and
the headings found on it. You never modify a document; the methods below return new ones.

```python
from foveate import Document

report = Document.load("annual_report.pdf")   # needs: pip install "foveate[pdf]"
report.id                    # "annual_report" (the file name; used in citations)
len(report.pages)            # 160
report.token_count           # 118,420
report.page(60).text         # the text of page 60
report.outline()             # [Heading(title="Cash flows", page=58), ...]
```

## Where pages come from

| Format | Pages |
|---|---|
| PDF | The real pages of the file. Blank pages are kept, so numbers match the printed document. |
| DOCX, HTML, Markdown, text | There are no real pages, so Foveate cuts the text into pseudo-pages of about 500 tokens at paragraph boundaries. Set `page_tokens=` to change the size. A form-feed character (`\f`) in text always starts a new page. |

```python
manual = Document.load("manual.md", page_tokens=600)
raw = Document.load(pdf_bytes, format="pdf", doc_id="q3_report")
```

Pass `doc_id` when you load bytes, or when two files share a name. Ids must be unique across the
documents you give to one `Foveator`, because they appear in every citation.

Text files are decoded as UTF-8, and Windows line endings are converted to `\n` so a page looks the
same on every machine.

## Picking pages yourself

You do not have to let Foveate choose. When you already know where to look, ask for pages directly.

```python
report.select("12-14,40")        # a new Document with only those pages
report.around(40, radius=2)      # pages 38 to 42
report.page(40)                  # one Page
report.text(markers=True)        # all text, each page prefixed with [annual_report p.N]
```

The `markers=True` form is what you want to paste into a prompt yourself: the markers let the model
cite pages.

## Limits to know about

* **Scanned PDFs have no text.** Foveate reads the text layer with `pypdf`. A scan without one yields
  empty pages. Run OCR first. (Scanned pages and images are on the [roadmap](../roadmap.md).)
* **Tables become text.** Row and column structure may be lost, which can make numeric questions
  harder. A number that appears in the extracted text can still be found and cited.
* **Page numbers are file page numbers**, not the numbers printed on the page. A report whose page
  "1" is the 7th sheet will cite the 7th.

## Next

[Selecting pages](selection.md) explains how Foveate decides which pages matter for a question.
