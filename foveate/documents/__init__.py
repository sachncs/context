"""Page-addressable documents and loaders.

Built in: text, Markdown and HTML (standard library only). PDF and DOCX are
plugin packages (`foveate-pdf`, `foveate-docx`) discovered automatically.
"""

from foveate.documents.document import Document, unique_ids
from foveate.documents.loaders import (
    HtmlLoader,
    Loader,
    MarkdownLoader,
    TextLoader,
)
from foveate.documents.page import Heading, Page, detect_headings, parse_pages

__all__ = [
    "Document",
    "Heading",
    "HtmlLoader",
    "Loader",
    "MarkdownLoader",
    "Page",
    "TextLoader",
    "detect_headings",
    "parse_pages",
    "unique_ids",
]
