"""Unit tests for text extraction and chunking (pure, no external services)."""

import pytest

from app.ingestion import chunk_text, extract_text


def test_extract_txt_decodes_bytes():
    text = extract_text("notes.txt", b"hello world")
    assert text == "hello world"


def test_extract_markdown_supported():
    text = extract_text("readme.md", b"# Title\n\nBody text.")
    assert "Title" in text and "Body text" in text


def test_extract_unsupported_type_raises():
    with pytest.raises(ValueError):
        extract_text("image.png", b"\x89PNG")


def test_chunk_empty_text_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("A short sentence.", chunk_size=900, overlap=150)
    assert chunks == ["A short sentence."]


def test_chunk_long_text_splits_with_overlap():
    # 60 words, ~ hundreds of chars -> multiple chunks at small chunk_size.
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    # Every chunk respects the size bound (allowing the snap-to-space slack).
    assert all(len(c) <= 120 for c in chunks)


def test_chunk_does_not_lose_content():
    text = " ".join(f"token{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # First and last tokens must appear somewhere in the chunk set.
    joined = " ".join(chunks)
    assert "token0" in joined
    assert "token99" in joined


def test_chunk_overlap_creates_shared_content():
    text = " ".join(f"w{i}" for i in range(80))
    chunks = chunk_text(text, chunk_size=100, overlap=40)
    assert len(chunks) >= 2
    # With overlap, consecutive chunks should share at least one token.
    first_tokens = set(chunks[0].split())
    second_tokens = set(chunks[1].split())
    assert first_tokens & second_tokens
