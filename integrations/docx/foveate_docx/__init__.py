"""Word (.docx) loader for Foveate (uses `python-docx`).

Installing this package is enough: Foveate discovers it through the
`foveate.loaders` entry point.

    from foveate import Document
    plan = Document.load("plan.docx")

Word has no stable page boundaries, so the text is cut into pseudo-pages of
about `page_tokens` tokens. Heading styles become `# ` headings.
"""

from __future__ import annotations

import dataclasses
import io

import docx

from foveate import errors
from foveate.documents import Loader
from foveate.documents import loaders as loaders_lib
from foveate.tokenizers import base as tokenizer_base


@Loader.registry.register("docx")
@dataclasses.dataclass(frozen=True)
class DocxLoader(Loader):
    """Reads Word documents with `python-docx`."""

    def extract(
        self, data: bytes, tokenizer: tokenizer_base.Tokenizer
    ) -> list[str]:
        try:
            parsed = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise errors.ValidationError(f"cannot read DOCX: {exc}") from exc
        lines = []
        for paragraph in parsed.paragraphs:
            style = (paragraph.style.name or "") if paragraph.style else ""
            prefix = "# " if style.startswith("Heading") else ""
            if paragraph.text.strip():
                lines.append(prefix + paragraph.text)
        return loaders_lib.paginate(
            "\n\n".join(lines), tokenizer, self.page_tokens
        )
