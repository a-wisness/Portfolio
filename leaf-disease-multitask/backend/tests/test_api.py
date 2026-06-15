"""API tests with stubbed predictor + registry — no TensorFlow required."""
from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.observability import metrics


class FakePredictor:
    classes = ["alpha_leaf", "beta_leaf", "gamma_leaf"]
    version = "v1"

    def __init__(self):
        self.reloaded = False

    def cache_stats(self):
        return {"size": 0, "capacity": 64, "hits": 0, "misses": 0}

    def reload(self):
        self.reloaded = True
        return self.version

    def predict_bytes(self, raw):
        if not raw or raw == b"not an image":
            raise ValueError("bad image")
        tiny = base64.b64encode(b"fake-png-bytes").decode()
        return {
            "predicted_class": "beta_leaf",
            "confidence": 0.91,
            "top_k": [
                {"label": "beta_leaf", "confidence": 0.91},
                {"label": "alpha_leaf", "confidence": 0.06},
            ],
            "leaf_coverage": 0.42,
            "mask_png_base64": tiny,
            "overlay_png_base64": tiny,
        }


class FakeRegistry:
    def __init__(self, settings=None):
        pass

    def active_version(self):
        return "v1"

    def list_versions(self):
        return [{"version": "v1", "active": True, "created_utc": "t",
                 "num_classes": 3, "val_metrics": {"loss": 0.9}}]

    def set_active(self, version):
        if version != "v1":
            raise FileNotFoundError(version)


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(main, "_predictor", FakePredictor())
    monkeypatch.setattr(main, "artifact_exists", lambda: True)
    monkeypatch.setattr(main, "Registry", FakeRegistry)


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(main, "_predictor", None)
    monkeypatch.setattr(main, "artifact_exists", lambda: False)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 120, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_without_model(client, no_model):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_with_model(client, stub_model):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert body["num_classes"] == 3
    assert body["active_version"] == "v1"
    assert body["cache"]["capacity"] == 64


def test_health_reports_registry_before_predictor_loaded(client, monkeypatch):
    # Artifact on disk but predictor not yet lazily loaded.
    monkeypatch.setattr(main, "_predictor", None)
    monkeypatch.setattr(main, "artifact_exists", lambda: True)
    monkeypatch.setattr(main, "Registry", FakeRegistry)
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert body["active_version"] == "v1"
    assert body["num_classes"] == 3
    assert body["cache"] is None          # not loaded yet


def test_classes(client, stub_model):
    r = client.get("/api/classes")
    assert r.status_code == 200
    assert r.json()["classes"] == FakePredictor.classes


def test_list_models(client, stub_model):
    body = client.get("/api/models").json()
    assert body["active_version"] == "v1"
    assert body["versions"][0]["version"] == "v1"
    assert body["versions"][0]["active"] is True


def test_activate_model(client, stub_model):
    r = client.post("/api/models/v1/activate")
    assert r.status_code == 200
    assert r.json()["active_version"] == "v1"
    assert main._predictor.reloaded is True


def test_activate_unknown_model_404(client, stub_model):
    r = client.post("/api/models/nope/activate")
    assert r.status_code == 404


def test_reload_model(client, stub_model):
    r = client.post("/api/models/reload")
    assert r.status_code == 200
    assert r.json()["active_version"] == "v1"


def test_predict_success(client, stub_model):
    r = client.post("/api/predict", files={"file": ("leaf.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "beta_leaf"
    assert body["confidence"] == pytest.approx(0.91)
    assert len(body["top_k"]) == 2


def test_metrics_endpoint_records_requests(client, stub_model):
    metrics.reset()
    client.get("/api/health")
    client.get("/api/classes")
    # The in-flight /api/metrics request is recorded only after this handler
    # returns, so it counts the two preceding requests.
    body = client.get("/api/metrics").json()
    assert body["requests"]["total"] == 2
    assert body["requests"]["by_status"]["2xx"] == 2
    assert body["latency_ms"]["count"] == 2


def test_metrics_records_prediction_distribution(client, stub_model):
    metrics.reset()
    client.post("/api/predict", files={"file": ("leaf.png", _png_bytes(), "image/png")})
    preds = client.get("/api/metrics").json()["predictions"]
    assert preds["total"] == 1
    assert preds["by_class"].get("beta_leaf") == 1


def test_predict_without_model_returns_503(client, no_model):
    r = client.post("/api/predict", files={"file": ("leaf.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


def test_predict_rejects_empty_file(client, stub_model):
    r = client.post("/api/predict", files={"file": ("leaf.png", b"", "image/png")})
    assert r.status_code == 400


def test_predict_rejects_non_image(client, stub_model):
    r = client.post("/api/predict", files={"file": ("note.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
