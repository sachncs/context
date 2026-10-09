"""A stand-in `pdf` loader so core tests need no PDF library.

The real loader is the `foveate-pdf` plugin (integrations/pdf), which is
tested on its own. Importing this module registers the stand-in unless a
plugin already provides `pdf`.
"""

import dataclasses

from foveate import errors
from foveate.documents import Loader
from foveate.tokenizers import base as tokenizer_base

MAGIC = b"%PDF-fake\n"


def make_pdf(pages: int) -> bytes:
    """Returns fake PDF bytes with one marker sentence per page."""
    body = "\f".join(f"Marker PAGEMARK{n:03d}." for n in range(1, pages + 1))
    return MAGIC + body.encode("utf-8")


if "pdf" not in Loader.registry.names():

    @Loader.registry.register("pdf")
    @dataclasses.dataclass(frozen=True)
    class FakePdfLoader(Loader):
        def extract(
            self, data: bytes, tokenizer: tokenizer_base.Tokenizer
        ) -> list[str]:
            if not data.startswith(MAGIC):
                raise errors.ValidationError("cannot read PDF")
            return data[len(MAGIC) :].decode("utf-8").split("\f")
