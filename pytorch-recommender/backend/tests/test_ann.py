"""Tests for the ANN retrieval abstraction."""

import importlib.util

import numpy as np
import pytest

from app.ann import BruteForceIndex, FaissIndex, build_index

faiss_available = importlib.util.find_spec("faiss") is not None


def _normalized(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _toy_embeddings():
    # Four items; items 0 and 1 point in nearly the same direction.
    return _normalized(np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    ))


def test_brute_force_returns_sorted_descending():
    idx = BruteForceIndex(_toy_embeddings())
    ids, scores = idx.search(_toy_embeddings()[0], k=4)
    assert ids[0] == 0                       # the query item itself is closest
    assert ids[1] == 1                       # its near-twin is next
    assert scores == sorted(scores, reverse=True)


def test_brute_force_caps_k_to_corpus():
    idx = BruteForceIndex(_toy_embeddings())
    ids, _ = idx.search(_toy_embeddings()[0], k=99)
    assert len(ids) == 4


def test_build_index_defaults_to_brute_force():
    idx = build_index(_toy_embeddings(), use_faiss=False)
    assert isinstance(idx, BruteForceIndex)


def test_build_index_falls_back_when_faiss_missing(monkeypatch):
    # Even if asked for FAISS, a failure to construct must fall back gracefully.
    if not faiss_available:
        idx = build_index(_toy_embeddings(), use_faiss=True)
        assert isinstance(idx, BruteForceIndex)


@pytest.mark.skipif(not faiss_available, reason="faiss not installed")
def test_faiss_matches_brute_force_top1():
    emb = _toy_embeddings()
    brute = BruteForceIndex(emb)
    fa = FaissIndex(emb)
    for q in range(len(emb)):
        assert brute.search(emb[q], 1)[0][0] == fa.search(emb[q], 1)[0][0]
