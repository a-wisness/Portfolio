"""FastAPI application — the HTTP surface for the semantic search engine.

Routes:
  POST /api/ingest   upload + index a document
  POST /api/search   ask a question, get a cited answer
  GET  /api/stats    indexed documents + chunk counts
  POST /api/reset    clear the index
  GET  /api/health   liveness probe
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import embeddings, ingestion, search, vectorstore
from .schemas import (
    DocumentInfo,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

app = FastAPI(
    title="Semantic Search Studio",
    description="LLM-powered semantic search over your documents.",
    version="1.0.0",
)

# Allow the Vite dev server to call the API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        text = ingestion.extract_text(file.filename or "upload", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = ingestion.chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No extractable text found in file."
        )

    vectors = embeddings.embed_texts(chunks)
    count = vectorstore.add_chunks(file.filename or "upload", chunks, vectors)

    return IngestResponse(
        filename=file.filename or "upload",
        chunks_indexed=count,
        message=f"Indexed {count} chunks from {file.filename!r}.",
    )


@app.post("/api/search", response_model=SearchResponse)
def run_search(req: SearchRequest) -> SearchResponse:
    _, total = vectorstore.stats()
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first.",
        )
    return search.answer_query(req.query, top_k=req.top_k)


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    counts, total = vectorstore.stats()
    docs = [
        DocumentInfo(filename=name, chunks=n) for name, n in sorted(counts.items())
    ]
    return StatsResponse(documents=docs, total_chunks=total)


@app.post("/api/reset")
def reset() -> dict[str, str]:
    vectorstore.reset()
    return {"status": "cleared"}
