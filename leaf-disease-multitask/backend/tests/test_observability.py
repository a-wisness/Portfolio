"""Metrics tests — no TensorFlow required."""
from __future__ import annotations

from app.observability import Metrics


def test_request_volume_and_status_buckets():
    m = Metrics()
    m.record_request("/api/predict", "POST", 200, 12.0)
    m.record_request("/api/predict", "POST", 200, 8.0)
    m.record_request("/api/predict", "POST", 400, 1.0)
    m.record_request("/api/health", "GET", 200, 0.5)
    snap = m.snapshot()
    assert snap["requests"]["total"] == 4
    assert snap["requests"]["by_status"]["2xx"] == 3
    assert snap["requests"]["by_status"]["4xx"] == 1
    assert snap["requests"]["by_path"]["POST /api/predict"] == 3


def test_latency_percentiles_ordered():
    m = Metrics()
    for v in range(1, 101):           # 1..100 ms
        m.record_request("/api/predict", "POST", 200, float(v))
    lat = m.snapshot()["latency_ms"]
    assert lat["count"] == 100
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]
    assert lat["max"] == 100.0
    assert 49 <= lat["p50"] <= 51


def test_prediction_distribution():
    m = Metrics()
    for label in ["Corn rust leaf", "Corn rust leaf", "Apple leaf"]:
        m.record_prediction(label)
    preds = m.snapshot()["predictions"]
    assert preds["total"] == 3
    assert preds["by_class"]["Corn rust leaf"] == 2
    assert preds["by_class"]["Apple leaf"] == 1


def test_latency_buffer_is_bounded():
    m = Metrics(max_samples=10)
    for v in range(50):
        m.record_request("/api/predict", "POST", 200, float(v))
    snap = m.snapshot()
    assert snap["requests"]["total"] == 50      # totals are exact
    assert snap["latency_ms"]["count"] == 10     # but samples are capped


def test_reset_clears_everything():
    m = Metrics()
    m.record_request("/api/health", "GET", 200, 1.0)
    m.record_prediction("Apple leaf")
    m.reset()
    snap = m.snapshot()
    assert snap["requests"]["total"] == 0
    assert snap["predictions"]["total"] == 0
    assert snap["latency_ms"]["count"] == 0
