"""Page-addressable documents and loaders."""

from foveate.documents.document import Document, unique_ids
from foveate.documents.loaders import (
    DocxLoader,
    HtmlLoader,
    Loader,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
)
from foveate.documents.page import Heading, Page, detect_headings, parse_pages

__all__ = [
    "Document",
    "DocxLoader",
    "Heading",
    "HtmlLoader",
    "Loader",
    "MarkdownLoader",
    "Page",
    "PdfLoader",
    "TextLoader",
    "detect_headings",
    "parse_pages",
    "unique_ids",
]
