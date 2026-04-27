from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def parse_docx(path: Path) -> str:
    """Extract plain text from a .docx (paragraphs and tables in document order)."""
    doc = Document(str(path))
    parts: list[str] = []
    for el in doc.element.body.iterchildren():
        if el.tag == qn("w:p"):
            p = Paragraph(el, doc)
            text = p.text.strip()
            if text:
                parts.append(text)
        elif el.tag == qn("w:tbl"):
            table = Table(el, doc)
            for row in table.rows:
                cells = [c.text.replace("\n", " ").strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n\n".join(parts)
