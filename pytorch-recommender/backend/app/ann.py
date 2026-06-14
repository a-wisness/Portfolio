"""Approximate-nearest-neighbor retrieval over item embeddings.

At MovieLens-100k/1M scale a brute-force cosine scan is already fast, but real
catalogs have millions of items where an ANN index is essential. This module
abstracts the lookup behind a small interface with two backends:

  * BruteForceIndex — exact cosine via a normalized matrix product (always
    available, no extra dependency).
  * FaissIndex      — a FAISS inner-product index (used only when faiss is
    installed and enabled in config).

Embeddings are assumed L2-normalized, so inner product == cosine similarity.
"""

from __future__ import annotations

import numpy as np


class BruteForceIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        sims = self.embeddings @ query.astype(np.float32)
        k = min(k, sims.shape[0])
        # Partial top-k, then sort just those k by score (descending).
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        return order.tolist(), sims[order].tolist()


class FaissIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        import faiss

        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        k = min(k, self.embeddings.shape[0])
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        scores, idx = self.index.search(q, k)
        return idx[0].tolist(), scores[0].tolist()


def build_index(embeddings: np.ndarray, use_faiss: bool = False):
    """Return a FAISS index if requested and importable, else brute force."""
    if use_faiss:
        try:
            return FaissIndex(embeddings)
        except Exception as exc:  # faiss missing or failed to build
            print(f"FAISS unavailable ({exc}); falling back to brute force.")
    return BruteForceIndex(embeddings)
