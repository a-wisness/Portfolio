"""FastAPI inference service for LeafLens.

Loads the active model artifact lazily on first use; the app starts (and
``/api/health`` responds) even before a model has been trained, reporting
``model_loaded: false`` so the stack degrades gracefully. Model versions can be
listed, activated, and hot-reloaded over HTTP without a restart.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from .config import get_settings
from .inference import LeafLensPredictor, artifact_exists
from .observability import metrics
from .registry import Registry
from .schemas import (
    ClassesResponse,
    HealthResponse,
    MetricsResponse,
    ModelsResponse,
    ModelVersion,
    PredictResponse,
    ReloadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("leaflens")

app = FastAPI(title="LeafLens", version="0.1.0")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Time + log every request and feed the in-process metrics."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    # Prefer the matched route template (e.g. /api/models/{version}/activate)
    # over the concrete URL so per-version paths don't blow up cardinality.
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    metrics.record_request(path, request.method, response.status_code, latency_ms)
    logger.info("%s %s -> %d (%.1f ms)", request.method, path,
                response.status_code, latency_ms)
    return response

# Module-level cache; tests can monkeypatch `main._predictor`.
_predictor: LeafLensPredictor | None = None


def get_predictor() -> LeafLensPredictor:
    global _predictor
    if _predictor is None:
        if not artifact_exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model available. Run `python -m app.train` first.",
            )
        _predictor = LeafLensPredictor()
    return _predictor


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _predictor is not None or artifact_exists()
    num_classes = active_version = cache = None
    if _predictor is not None:
        num_classes = len(_predictor.classes)
        active_version = _predictor.version
        cache = _predictor.cache_stats()
    elif loaded:
        # Report from the registry without paying the TF model-load cost.
        registry = Registry(get_settings())
        active_version = registry.active_version()
        active = next((v for v in registry.list_versions() if v["active"]), None)
        num_classes = active.get("num_classes") if active else None
    return HealthResponse(
        status="ok", model_loaded=loaded, num_classes=num_classes,
        active_version=active_version, cache=cache,
    )


@app.get("/api/classes", response_model=ClassesResponse)
def classes() -> ClassesResponse:
    return ClassesResponse(classes=get_predictor().classes)


@app.get("/api/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    registry = Registry(get_settings())
    versions = [ModelVersion(**v) for v in registry.list_versions()]
    return ModelsResponse(active_version=registry.active_version(), versions=versions)


@app.post("/api/models/{version}/activate", response_model=ReloadResponse)
def activate_model(version: str) -> ReloadResponse:
    registry = Registry(get_settings())
    try:
        registry.set_active(version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown model version: {version}")
    return ReloadResponse(active_version=get_predictor().reload())


@app.post("/api/models/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    return ReloadResponse(active_version=get_predictor().reload())


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    return MetricsResponse(**metrics.snapshot())


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        result = get_predictor().predict_bytes(raw)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")
    metrics.record_prediction(result["predicted_class"])
    return PredictResponse(**result)
