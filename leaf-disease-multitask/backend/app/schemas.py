"""Pydantic response models for the API. TensorFlow-free."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[ClassScore]
    leaf_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of pixels the segmentation head marked as leaf.",
    )
    mask_png_base64: str = Field(
        description="Predicted binary leaf mask as a base64-encoded PNG.",
    )
    overlay_png_base64: str = Field(
        description="Input image with the predicted mask overlaid, base64 PNG.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int | None = None
    active_version: str | None = None
    cache: dict | None = None


class ClassesResponse(BaseModel):
    classes: list[str]


class ModelVersion(BaseModel):
    version: str
    active: bool
    created_utc: str | None = None
    num_classes: int | None = None
    val_metrics: dict | None = None


class ModelsResponse(BaseModel):
    active_version: str | None = None
    versions: list[ModelVersion]


class ReloadResponse(BaseModel):
    active_version: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    requests: dict
    latency_ms: dict
    predictions: dict
