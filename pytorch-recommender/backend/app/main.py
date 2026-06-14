"""FastAPI application — serves recommendations from the active model version.

Recommendation routes:
  GET  /api/health                          liveness + active version
  GET  /api/movies?search=&limit=           browse / search the catalog
  POST /api/recommend                       cold-start recs from liked movies
  GET  /api/users/{user_id}/recommendations NeuMF recs for a known user
  GET  /api/movies/{movie_id}/similar       nearest movies in embedding space

Operational routes:
  GET  /api/models                          list trained versions
  POST /api/models/{version}/activate       hot-swap the served version
  POST /api/models/reload                   reload the active version from disk
  GET  /api/metrics                         serving metrics (latency, coverage)
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from . import registry
from .observability import collector
from .recommender import (
    ModelNotLoaded,
    UnknownUser,
    get_recommender,
    reload_recommender,
)
from .schemas import (
    HealthResponse,
    ModelVersion,
    Movie,
    ReloadResponse,
    RecommendRequest,
    RecommendResponse,
)

logger = logging.getLogger("cinematch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="CineMatch — PyTorch NCF Recommender",
    description="Movie recommendations from a Neural Collaborative Filtering model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # route.path gives the pattern (/api/users/{user_id}/…) so metrics aggregate by endpoint, not per-ID.
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    collector.record_request(path, response.status_code, elapsed_ms)
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# --------------------------------------------------------------------------- #
# Recommendation routes
# --------------------------------------------------------------------------- #
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rec = get_recommender()
    return HealthResponse(
        status="ok",
        model_loaded=rec.loaded,
        version=rec.version if rec.loaded else None,
        created_at=rec.created_at if rec.loaded else None,
        num_users=rec.num_users if rec.loaded else None,
        num_items=rec.num_items if rec.loaded else None,
        metrics=rec.metrics if rec.loaded else None,
    )


@app.get("/api/movies", response_model=list[Movie])
def list_movies(
    search: str = Query("", description="Case-insensitive title substring"),
    limit: int = Query(20, ge=1, le=100),
) -> list[Movie]:
    rec = _loaded()
    return [Movie(**m) for m in rec.search_movies(search, limit)]


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    rec = _loaded()
    k = req.top_k or 10
    results = rec.recommend_for_likes(req.liked_movie_ids, k)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="None of the provided movie IDs are in the catalog.",
        )
    collector.record_recommendations([r["movie_id"] for r in results])
    return RecommendResponse(
        recommendations=results,
        strategy="item-embedding similarity (cold-start)",
    )


@app.get("/api/users/{user_id}/recommendations", response_model=RecommendResponse)
def user_recommendations(
    user_id: int, top_k: int = Query(10, ge=1, le=100)
) -> RecommendResponse:
    rec = _loaded()
    try:
        results = rec.recommend_for_user(user_id, top_k)
    except UnknownUser as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    collector.record_recommendations([r["movie_id"] for r in results])
    return RecommendResponse(
        recommendations=results,
        strategy="neural collaborative filtering",
    )


@app.get("/api/movies/{movie_id}/similar", response_model=RecommendResponse)
def similar(movie_id: int, top_k: int = Query(10, ge=1, le=100)) -> RecommendResponse:
    rec = _loaded()
    results = rec.similar_movies(movie_id, top_k)
    if not results:
        raise HTTPException(status_code=404, detail="Unknown movie id.")
    collector.record_recommendations([r["movie_id"] for r in results])
    return RecommendResponse(
        recommendations=results,
        strategy="item-embedding nearest neighbors",
    )


# --------------------------------------------------------------------------- #
# Operational routes
# --------------------------------------------------------------------------- #
@app.get("/api/models", response_model=list[ModelVersion])
def list_models() -> list[ModelVersion]:
    return [ModelVersion(**v) for v in registry.list_versions()]


@app.post("/api/models/{version}/activate", response_model=ReloadResponse)
def activate_model(version: str) -> ReloadResponse:
    try:
        registry.set_active(version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rec = reload_recommender()
    return ReloadResponse(active_version=rec.version, model_loaded=rec.loaded)


@app.post("/api/models/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    rec = reload_recommender()
    return ReloadResponse(active_version=rec.version, model_loaded=rec.loaded)


@app.get("/api/metrics")
def metrics() -> dict:
    rec = get_recommender()
    return collector.snapshot(catalog_size=rec.num_items if rec.loaded else 0)


def _loaded():
    """Return the recommender, or 503 if no model artifact has been trained."""
    rec = get_recommender()
    try:
        rec._require_loaded()
    except ModelNotLoaded as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m app.train",
        ) from exc
    return rec
