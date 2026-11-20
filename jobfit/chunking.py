"""Split long text into overlapping chunks for retrieval (RAG)."""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into ~chunk_size character chunks, preferring paragraph/sentence boundaries.

    Overlap carries the tail of the previous chunk forward so context isn't lost at edges.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    # Paragraphs first, then sentences for paragraphs that are too long.
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n", para):
            sent = sent.strip()
            # Hard-split anything still longer than a chunk.
            while len(sent) > chunk_size:
                pieces.append(sent[:chunk_size])
                sent = sent[chunk_size:]
            if sent:
                pieces.append(sent)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n{piece}".strip() if tail else piece
        if len(current) > chunk_size:  # overlap pushed it over; drop the overlap
            current = piece
    if current:
        chunks.append(current)
    return chunks
