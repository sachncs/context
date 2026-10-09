import docx

from foveate import Document
from foveate.documents import Loader


def test_docx_is_discovered_and_keeps_headings(tmp_path):
    d = docx.Document()
    d.add_heading("Quarterly Plan", level=1)
    d.add_paragraph("We will ship the feature in March.")
    path = tmp_path / "plan.docx"
    d.save(path)
    doc = Document.load(path)
    assert "ship the feature" in doc.text()
    assert doc.outline()[0].title == "Quarterly Plan"
    assert "docx" in Loader.registry.names()
