"""Inference cache + reload tests — no TensorFlow (fake model injected)."""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from app.config import Settings
from app.inference import LeafLensPredictor
from app.registry import Registry


class FakeModel:
    def __init__(self, h, w, c):
        self.h, self.w, self.c = h, w, c
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        return {
            "segmentation": np.zeros((1, self.h, self.w, 1), np.float32),
            "classification": np.eye(self.c, dtype=np.float32)[None, 1],  # class 1
        }


def _make_predictor(tmp_path, cache_size=8) -> LeafLensPredictor:
    settings = Settings(artifacts_dir=tmp_path / "artifacts",
                        img_height=32, img_width=32, cache_size=cache_size,
                        top_k=2)
    reg = Registry(settings)
    _, vdir = reg.create_version("v1")
    (vdir / "model.keras").write_bytes(b"fake")
    reg.finalize("v1", ["leaf_a", "leaf_b", "leaf_c"], {"num_classes": 3})

    predictor = LeafLensPredictor(settings)
    predictor._model = FakeModel(32, 32, 3)   # inject; bypass TF load
    return predictor


def _png(color=(10, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_bytes_caches_identical_input(tmp_path):
    p = _make_predictor(tmp_path)
    raw = _png()
    r1 = p.predict_bytes(raw)
    r2 = p.predict_bytes(raw)
    assert r1["predicted_class"] == "leaf_b"        # class 1 one-hot
    assert r2 == r1
    assert p._model.calls == 1                       # second served from cache
    stats = p.cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_distinct_inputs_miss(tmp_path):
    p = _make_predictor(tmp_path)
    p.predict_bytes(_png((10, 10, 10)))
    p.predict_bytes(_png((250, 250, 250)))
    assert p._model.calls == 2
    assert p.cache_stats()["misses"] == 2


def test_cache_eviction_bounded(tmp_path):
    p = _make_predictor(tmp_path, cache_size=2)
    for i in range(5):
        p.predict_bytes(_png((i, i, i)))
    assert p.cache_stats()["size"] <= 2


def test_reload_clears_cache(tmp_path):
    p = _make_predictor(tmp_path)
    p.predict_bytes(_png())
    assert p.cache_stats()["size"] == 1
    fake = p._model
    p.reload()
    assert p.cache_stats()["size"] == 0
    assert p.version == "v1"
    # reload drops the loaded model so it would be re-fetched lazily.
    assert p._model is None or p._model is not fake
