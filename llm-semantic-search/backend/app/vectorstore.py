"""ChromaDB persistent vector store wrapper.

We supply our own (sentence-transformers) embeddings, so Chroma is used purely
as a cosine-similarity index with metadata. The store survives restarts via
on-disk persistence.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

import chromadb

from .config import settings


@lru_cache(maxsize=1)
def _collection() -> "chromadb.Collection":
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    filename: str, chunks: list[str], embeddings: list[list[float]]
) -> int:
    """Index a document's chunks. Returns the number of chunks added."""
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"filename": filename, "chunk_index": i} for i in range(len(chunks))
    ]
    _collection().add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def query(embedding: list[float], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks as dicts with text + metadata."""
    res = _collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits = []
    for text, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": text,
                "filename": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                # cosine distance -> similarity score
                "score": round(1.0 - float(dist), 4),
            }
        )
    return hits


def stats() -> tuple[dict[str, int], int]:
    """Return (per-document chunk counts, total chunk count)."""
    col = _collection()
    total = col.count()
    if total == 0:
        return {}, 0
    res = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in res["metadatas"]:
        name = meta.get("filename", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts, total


def reset() -> None:
    """Drop all indexed data (used by the clear endpoint)."""
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    client.delete_collection(settings.collection_name)
    _collection.cache_clear()
