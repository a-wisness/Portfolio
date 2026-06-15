"""Dependency-free, in-process metrics.

A single thread-safe ``Metrics`` instance accumulates request volume, latency
percentiles, status-code buckets, and the distribution of predicted classes.
No external metrics backend — just a snapshot served at ``/api/metrics``. Latency
samples are kept in a bounded ring buffer so memory stays flat.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque


class Metrics:
    def __init__(self, max_samples: int = 2048):
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._start = time.time()
            self._total = 0
            self._by_path: Counter[str] = Counter()
            self._by_status: Counter[str] = Counter()
            self._latencies: deque[float] = deque(maxlen=self._max_samples)
            self._pred: Counter[str] = Counter()
            self._pred_total = 0

    def record_request(self, path: str, method: str, status: int, latency_ms: float) -> None:
        bucket = f"{status // 100}xx"
        with self._lock:
            self._total += 1
            self._by_path[f"{method} {path}"] += 1
            self._by_status[bucket] += 1
            self._latencies.append(latency_ms)

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self._pred[label] += 1
            self._pred_total += 1

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * pct / 100.0
        lo = int(k)
        hi = min(lo + 1, len(sorted_vals) - 1)
        if lo == hi:
            return round(sorted_vals[lo], 2)
        return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo), 2)

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            latency = {
                "count": len(lat),
                "p50": self._percentile(lat, 50),
                "p95": self._percentile(lat, 95),
                "p99": self._percentile(lat, 99),
                "max": round(lat[-1], 2) if lat else 0.0,
                "avg": round(sum(lat) / len(lat), 2) if lat else 0.0,
            }
            return {
                "uptime_seconds": round(time.time() - self._start, 1),
                "requests": {
                    "total": self._total,
                    "by_path": dict(self._by_path),
                    "by_status": dict(self._by_status),
                },
                "latency_ms": latency,
                "predictions": {
                    "total": self._pred_total,
                    "by_class": dict(self._pred),
                },
            }


# Process-wide singleton.
metrics = Metrics()
