"""Document text extraction and chunking.

Turns an uploaded file (PDF / Markdown / plain text) into a list of overlapping
text chunks suitable for embedding. Overlap preserves context that would
otherwise be split across a chunk boundary.
"""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from .config import settings


def extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded file based on its extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(raw)
    if lower.endswith((".txt", ".md", ".markdown")):
        return raw.decode("utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file type: {filename!r}. Use PDF, .txt, or .md."
    )


def _extract_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _normalize(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks readable.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks, trying to break on whitespace.

    A simple, dependency-free chunker: walk the string in windows of
    ``chunk_size`` characters with ``overlap`` characters of carry-over,
    snapping each cut back to the nearest space so words aren't sliced.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = _normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Snap to a nearby space to avoid cutting mid-word (unless at the end).
        if end < n:
            space = text.rfind(" ", start + chunk_size - overlap, end)
            if space != -1 and space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
