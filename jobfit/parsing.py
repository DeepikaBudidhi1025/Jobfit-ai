"""Extract plain text from uploaded resumes (PDF, DOCX, TXT/MD)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO, Union

FileLike = Union[str, Path, bytes, BinaryIO]


class UnsupportedFileError(ValueError):
    """Raised when a file type cannot be parsed."""


def _read_bytes(source: FileLike) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    return data if isinstance(data, bytes) else data.encode("utf-8")


def _pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _docx_to_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_text(source: FileLike, filename: str | None = None) -> str:
    """Return cleaned text from a PDF, DOCX or plain-text file."""
    if filename is None and isinstance(source, (str, Path)):
        filename = str(source)
    if filename is None:
        filename = getattr(source, "name", "")
    suffix = Path(filename or "").suffix.lower()
    data = _read_bytes(source)

    if suffix == ".pdf":
        text = _pdf_to_text(data)
    elif suffix == ".docx":
        text = _docx_to_text(data)
    elif suffix in {".txt", ".md", ""}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise UnsupportedFileError(f"Unsupported file type: {suffix}. Use PDF, DOCX or TXT.")

    text = clean_text(text)
    if not text:
        raise UnsupportedFileError(
            "No text found. If this is a scanned PDF, export it as a text-based PDF first."
        )
    return text
