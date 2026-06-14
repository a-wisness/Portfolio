"""Search orchestration: embed query -> retrieve -> synthesize.

This is the RAG pipeline that ties the layers together. Kept deliberately thin
so each step (embedding, retrieval, generation) stays independently testable.
"""

from __future__ import annotations

from . import embeddings, llm, vectorstore
from .config import settings
from .schemas import SearchResponse, Source


def answer_query(query: str, top_k: int | None = None) -> SearchResponse:
    k = top_k or settings.top_k

    query_vec = embeddings.embed_query(query)
    hits = vectorstore.query(query_vec, top_k=k)

    answer = llm.synthesize_answer(query, hits)

    sources = [
        Source(
            index=i,
            filename=hit["filename"],
            chunk_index=hit["chunk_index"],
            text=hit["text"],
            score=hit["score"],
        )
        for i, hit in enumerate(hits, start=1)
    ]

    return SearchResponse(query=query, answer=answer, sources=sources)
