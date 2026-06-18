"""Search orchestration: embed query -> retrieve -> synthesize.

This is the RAG pipeline that ties the layers together. Kept deliberately thin
so each step (embedding, retrieval, generation) stays independently testable.
It is also the natural place to instrument the pipeline: each stage is timed and
token usage is captured so the response carries per-query performance metrics.
"""

from __future__ import annotations

import time

from . import embeddings, llm, vectorstore
from .config import settings
from .schemas import Metrics, SearchResponse, Source


def answer_query(query: str, top_k: int | None = None) -> SearchResponse:
    k = top_k or settings.top_k

    t0 = time.perf_counter()
    query_vec = embeddings.embed_query(query)
    t1 = time.perf_counter()
    hits = vectorstore.query(query_vec, top_k=k)
    t2 = time.perf_counter()
    answer, usage = llm.synthesize_answer(query, hits)
    t3 = time.perf_counter()

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

    cost = (
        usage["input_tokens"] / 1_000_000 * settings.input_price_per_mtok
        + usage["output_tokens"] / 1_000_000 * settings.output_price_per_mtok
    )
    metrics = Metrics(
        embed_ms=round((t1 - t0) * 1000, 1),
        retrieve_ms=round((t2 - t1) * 1000, 1),
        synthesize_ms=round((t3 - t2) * 1000, 1),
        total_ms=round((t3 - t0) * 1000, 1),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        estimated_cost_usd=round(cost, 6),
        top_score=hits[0]["score"] if hits else None,
    )

    return SearchResponse(
        query=query, answer=answer, sources=sources, metrics=metrics
    )
