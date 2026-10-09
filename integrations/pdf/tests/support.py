"""Builds PDFs for the tests."""

from fpdf import FPDF


def make_pdf(pages: int, blank: tuple[int, ...] = ()) -> bytes:
    """Returns a PDF with a heading and a marker sentence on each page."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for number in range(1, pages + 1):
        pdf.add_page()
        if number in blank:
            continue
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(
            0, 10, f"SECTION {number} OVERVIEW", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0, 6, f"Marker PAGEMARK{number:03d}. " + "Filler text. " * 30
        )
    return bytes(pdf.output())
