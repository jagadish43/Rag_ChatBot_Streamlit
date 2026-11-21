"""Extract text and source labels from uploaded documents."""
import io
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass(frozen=True)
class Excerpt:
    source: str
    text: str


def extract(name, data):
    suffix = Path(name).suffix.lower()
    stream = io.BytesIO(data)
    if suffix == ".pdf":
        return [
            Excerpt(f"{name}, page {number}", page.extract_text() or "")
            for number, page in enumerate(PdfReader(stream).pages, 1)
        ]
    if suffix == ".docx":
        document = Document(stream)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        paragraphs.extend(
            " | ".join(cell.text for cell in row.cells)
            for table in document.tables for row in table.rows
        )
        return [Excerpt(name, "\n".join(paragraphs))]
    return [Excerpt(name, data.decode("utf-8-sig"))]
