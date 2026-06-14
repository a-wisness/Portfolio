"""Pydantic request/response models — the typed API contract."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int | None = Field(
        None, ge=1, le=20, description="Override how many passages to retrieve"
    )


class Source(BaseModel):
    """One retrieved passage backing the answer."""

    index: int = Field(..., description="Citation number referenced as [index]")
    filename: str
    chunk_index: int
    text: str
    score: float = Field(..., description="Cosine similarity (higher = closer)")


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class StatsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
