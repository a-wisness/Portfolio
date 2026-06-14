"""Lightweight in-process serving metrics.

No external dependency (no Prometheus client) — just thread-safe counters and a
bounded latency window, exposed via /api/metrics. Tracks request volume,
latency percentiles, status codes, and recommendation **coverage** (the fraction
of the catalog the system has ever recommended — a useful health signal, since a
recommender that only ever surfaces a handful of popular items has a problem).
"""

from __future__ import annotations

import threading
import time
from collections import deque


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class MetricsCollector:
    def __init__(self, max_latencies: int = 1000) -> None:
        self._lock = threading.Lock()
        self._start = time.time()
        self.total_requests = 0
        self.requests_by_path: dict[str, int] = {}
        self.status_counts: dict[str, int] = {}
        self.latencies_ms: deque[float] = deque(maxlen=max_latencies)
        self.recommendation_calls = 0
        self.recommended_item_ids: set[int] = set()

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.requests_by_path[path] = self.requests_by_path.get(path, 0) + 1
            bucket = f"{status_code // 100}xx"
            self.status_counts[bucket] = self.status_counts.get(bucket, 0) + 1
            self.latencies_ms.append(latency_ms)

    def record_recommendations(self, movie_ids: list[int]) -> None:
        with self._lock:
            self.recommendation_calls += 1
            self.recommended_item_ids.update(movie_ids)

    def snapshot(self, catalog_size: int) -> dict:
        with self._lock:
            lat = sorted(self.latencies_ms)
            covered = len(self.recommended_item_ids)
            return {
                "uptime_seconds": round(time.time() - self._start, 1),
                "total_requests": self.total_requests,
                "requests_by_path": dict(self.requests_by_path),
                "status_counts": dict(self.status_counts),
                "latency_ms": {
                    "count": len(lat),
                    "avg": round(sum(lat) / len(lat), 2) if lat else 0.0,
                    "p50": round(_percentile(lat, 0.50), 2),
                    "p95": round(_percentile(lat, 0.95), 2),
                    "max": round(max(lat), 2) if lat else 0.0,
                },
                "recommendations": {
                    "calls": self.recommendation_calls,
                    "distinct_items_recommended": covered,
                    "catalog_size": catalog_size,
                    "coverage": round(covered / catalog_size, 4) if catalog_size else 0.0,
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._start = time.time()
            self.total_requests = 0
            self.requests_by_path.clear()
            self.status_counts.clear()
            self.latencies_ms.clear()
            self.recommendation_calls = 0
            self.recommended_item_ids.clear()


collector = MetricsCollector()
