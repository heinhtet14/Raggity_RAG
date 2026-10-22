"""Step 1 of ingestion: turn an uploaded file into plain text, one Section per page.

Keeping the page number lets an answer cite "report.pdf p.4" instead of just "report.pdf".
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None  # 1-based; None for formats without pages


class UnsupportedFileType(ValueError):
    pass


def load(filename: str, data: bytes) -> list[Section]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _load_pdf(data)
    if name.endswith((".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html")):
        return [Section(data.decode("utf-8", errors="replace"))]
    raise UnsupportedFileType(f"don't know how to read {filename!r}")


def _load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text, page=number))
    # A scanned PDF has no text layer, so this is empty. OCR is a roadmap item.
    return sections
