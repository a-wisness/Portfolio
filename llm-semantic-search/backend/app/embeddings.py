"""Local embedding model wrapper (sentence-transformers).

The model (~90 MB) is downloaded and cached on first use, then loaded lazily so
the API starts fast and only pays the load cost when embeddings are first
needed. Embeddings run on CPU with no API key or per-call cost.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so the app (and the test suite) can load without the
    # heavy ML stack until embeddings are actually computed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents/chunks."""
    if not texts:
        return []
    vectors = _model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
