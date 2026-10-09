"""PDF loader for Foveate (uses `pypdf`).

Installing this package is enough: Foveate discovers it through the
`foveate.loaders` entry point, so `Document.load("report.pdf")` just works.

    from foveate import Document
    report = Document.load("report.pdf")
    report = Document.load("report.pdf", layout=True)   # keep table layout

PDFs keep their real page boundaries, so page numbers match the file. A PDF
without a text layer (a scan) yields empty pages; run OCR first.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Literal

import pypdf

from foveate import errors
from foveate.documents import Loader
from foveate.tokenizers import base as tokenizer_base


@Loader.registry.register("pdf")
@dataclasses.dataclass(frozen=True)
class PdfLoader(Loader):
    """Reads PDFs with `pypdf`.

    Attributes:
        layout: Use pypdf's layout-preserving extraction (better tables,
            slower).
    """

    layout: bool = False

    def extract(
        self, data: bytes, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise errors.ValidationError("PDF is password protected")
            mode: Literal["plain", "layout"] = (
                "layout" if self.layout else "plain"
            )
            return [
                page.extract_text(extraction_mode=mode) or ""
                for page in reader.pages
            ]
        except errors.FoveateError:
            raise
        except Exception as exc:
            raise errors.ValidationError(f"cannot read PDF: {exc}") from exc
