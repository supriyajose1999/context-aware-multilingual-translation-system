"""Extract plain text from uploaded documents (PDF, DOCX, TXT)."""

from __future__ import annotations

import os


def extract_text(filepath: str) -> str:
    """Extract text from a PDF, DOCX, or TXT file based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(filepath)
    if ext == ".docx":
        return _extract_docx(filepath)
    if ext == ".txt":
        return _extract_txt(filepath)
    raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .docx, or .txt")


def _extract_pdf(filepath: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _extract_docx(filepath: str) -> str:
    from docx import Document
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
