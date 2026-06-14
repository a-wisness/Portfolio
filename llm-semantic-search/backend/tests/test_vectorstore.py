"""Integration tests for the ChromaDB vector-store wrapper.

Uses the deterministic fake embeddings from conftest against a real (temp)
ChromaDB instance.
"""

from app import embeddings, vectorstore


def _index(filename: str, chunks: list[str]) -> int:
    vectors = embeddings.embed_texts(chunks)
    return vectorstore.add_chunks(filename, chunks, vectors)


def test_add_and_count():
    added = _index("doc.txt", ["alpha chunk", "beta chunk", "gamma chunk"])
    assert added == 3
    counts, total = vectorstore.stats()
    assert total == 3
    assert counts == {"doc.txt": 3}


def test_query_returns_best_match_first():
    chunks = ["the cat sat on the mat", "quantum entanglement physics", "baking bread"]
    _index("mixed.txt", chunks)

    # Query identical to a stored chunk -> that chunk should rank first
    # (cosine similarity ~1.0 for matching fake vectors).
    q_vec = embeddings.embed_query("quantum entanglement physics")
    hits = vectorstore.query(q_vec, top_k=3)

    assert len(hits) == 3
    assert hits[0]["text"] == "quantum entanglement physics"
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_query_respects_top_k():
    _index("many.txt", [f"chunk number {i}" for i in range(10)])
    hits = vectorstore.query(embeddings.embed_query("chunk number 3"), top_k=4)
    assert len(hits) == 4


def test_hit_carries_metadata():
    _index("meta.txt", ["only chunk"])
    hits = vectorstore.query(embeddings.embed_query("only chunk"), top_k=1)
    hit = hits[0]
    assert hit["filename"] == "meta.txt"
    assert hit["chunk_index"] == 0
    assert -1.0 <= hit["score"] <= 1.0


def test_reset_clears_everything():
    _index("doc.txt", ["a", "b"])
    vectorstore.reset()
    counts, total = vectorstore.stats()
    assert total == 0
    assert counts == {}


def test_stats_aggregates_multiple_documents():
    _index("one.txt", ["x", "y"])
    _index("two.txt", ["z"])
    counts, total = vectorstore.stats()
    assert total == 3
    assert counts == {"one.txt": 2, "two.txt": 1}
