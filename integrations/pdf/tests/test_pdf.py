import pytest
from support import make_pdf

from foveate import Document, errors
from foveate.documents import Loader
from foveate.tokenizers import HeuristicTokenizer

TOK = HeuristicTokenizer()


def test_the_plugin_is_discovered_without_importing_it(tmp_path):
    path = tmp_path / "filing.pdf"
    path.write_bytes(make_pdf(3))
    assert Document.load(path, tokenizer=TOK).id == "filing"
    assert "pdf" in Loader.registry.names()


def test_pdf_has_real_pages_and_headings(tmp_path):
    path = tmp_path / "filing.pdf"
    path.write_bytes(make_pdf(12, blank=(5,)))
    doc = Document.load(path, tokenizer=TOK)
    assert doc.id == "filing" and len(doc.pages) == 12
    assert "PAGEMARK007" in doc.page(7).text
    assert "PAGEMARK007" not in doc.page(8).text
    assert doc.metadata["empty_pages"] == "1" and not doc.page(5).text.strip()
    assert any("SECTION 3" in h for h in doc.page(3).headings)
    assert doc.select("7").text().count("PAGEMARK") == 1


def test_bytes_and_layout_option():
    doc = Document.load(make_pdf(2), format="pdf", doc_id="b", layout=True)
    assert doc.id == "b" and len(doc.pages) == 2


def test_corrupt_and_missing_files(tmp_path):
    with pytest.raises(errors.ValidationError):
        Document.load(b"%PDF-1.4 not really", format="pdf")
    with pytest.raises(errors.ValidationError):
        Document.load(tmp_path / "missing.pdf")
