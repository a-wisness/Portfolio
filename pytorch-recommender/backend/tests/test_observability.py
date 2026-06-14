"""Tests for the in-process serving metrics collector."""

from app.observability import MetricsCollector, _percentile


def test_percentile_basic():
    vals = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile(vals, 0.0) == 0.0
    assert _percentile(vals, 1.0) == 40.0
    assert _percentile(vals, 0.5) == 20.0


def test_percentile_empty():
    assert _percentile([], 0.95) == 0.0


def test_records_requests_and_latency():
    c = MetricsCollector()
    c.record_request("/api/health", 200, 5.0)
    c.record_request("/api/health", 200, 15.0)
    c.record_request("/api/movies", 503, 2.0)
    snap = c.snapshot(catalog_size=100)
    assert snap["total_requests"] == 3
    assert snap["requests_by_path"]["/api/health"] == 2
    assert snap["status_counts"]["2xx"] == 2
    assert snap["status_counts"]["5xx"] == 1
    assert snap["latency_ms"]["count"] == 3
    assert snap["latency_ms"]["max"] == 15.0


def test_recommendation_coverage():
    c = MetricsCollector()
    c.record_recommendations([1, 2, 3])
    c.record_recommendations([3, 4])  # 3 repeats -> 4 distinct total
    snap = c.snapshot(catalog_size=8)
    assert snap["recommendations"]["calls"] == 2
    assert snap["recommendations"]["distinct_items_recommended"] == 4
    assert snap["recommendations"]["coverage"] == 0.5  # 4 / 8


def test_coverage_zero_when_catalog_unknown():
    c = MetricsCollector()
    c.record_recommendations([1, 2])
    assert c.snapshot(catalog_size=0)["recommendations"]["coverage"] == 0.0


def test_reset_clears_state():
    c = MetricsCollector()
    c.record_request("/x", 200, 1.0)
    c.record_recommendations([1])
    c.reset()
    snap = c.snapshot(catalog_size=10)
    assert snap["total_requests"] == 0
    assert snap["recommendations"]["distinct_items_recommended"] == 0
