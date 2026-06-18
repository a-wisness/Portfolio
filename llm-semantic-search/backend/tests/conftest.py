"""Shared test fixtures.

The suite runs fully offline and fast by:
  * pointing ChromaDB at a fresh temp directory per test (real vector-store
    integration, but isolated and disposable),
  * replacing the local embedding model with a deterministic fake (no ~90 MB
    download, no CPU cost),
  * stubbing the Claude call (no API key, no network, no token spend).

This lets us exercise the real ingestion -> chunk -> index -> retrieve -> answer
pipeline and the FastAPI routes without any external dependency.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from app import embeddings, llm, vectorstore
from app.config import settings

_DIM = 16


def _fake_vector(text: str) -> list[float]:
    """Deterministic, normalized pseudo-embedding derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:_DIM]
    vec = [b / 255.0 for b in digest]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Use a clean ChromaDB directory for every test."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    vectorstore._collection.cache_clear()
    yield
    vectorstore._collection.cache_clear()


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    """Swap the heavy embedding model and the Claude call for test doubles."""
    monkeypatch.setattr(
        embeddings, "embed_texts", lambda texts: [_fake_vector(t) for t in texts]
    )
    monkeypatch.setattr(
        embeddings, "embed_query", lambda q: _fake_vector(q)
    )

    def fake_synthesize(query: str, passages: list[dict]) -> tuple[str, dict]:
        usage = {"input_tokens": 42, "output_tokens": 7}
        if not passages:
            return "The documents don't appear to cover this.", {
                "input_tokens": 0,
                "output_tokens": 0,
            }
        return f"Synthesized answer for {query!r} grounded in [1].", usage

    monkeypatch.setattr(llm, "synthesize_answer", fake_synthesize)
