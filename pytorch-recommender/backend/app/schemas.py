"""Pydantic request/response models — the typed API contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: list[str]


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float = Field(..., description="Model score (higher = stronger match)")


class RecommendRequest(BaseModel):
    liked_movie_ids: list[int] = Field(
        ..., min_length=1, description="Movie IDs the user likes"
    )
    top_k: int | None = Field(None, ge=1, le=100)


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    strategy: str = Field(..., description="How the recommendations were produced")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str | None = None
    created_at: str | None = None
    num_users: int | None = None
    num_items: int | None = None
    metrics: dict[str, float] | None = None


class ModelVersion(BaseModel):
    version: str
    active: bool = False
    created_at: str | None = None
    dataset: str | None = None
    num_users: int | None = None
    num_items: int | None = None
    metrics: dict[str, float] | None = None


class ReloadResponse(BaseModel):
    active_version: str | None
    model_loaded: bool
