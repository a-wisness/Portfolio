"""Load a trained artifact (via the registry) and run single-image prediction.

Heavy deps (TensorFlow) are imported lazily inside methods so importing this
module (and the API) stays light and test-friendly. Predictions are served from
a bounded LRU cache keyed by image content + model version.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from collections import OrderedDict

import numpy as np
from PIL import Image

from .config import Settings, get_settings
from .registry import Registry


def _png_base64(arr: np.ndarray) -> str:
    """Encode a uint8 HxW (or HxWxC) array as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a green tint over leaf pixels of ``image`` (both uint8, same HxW)."""
    out = image.astype(np.float32)
    green = np.array([40, 220, 80], dtype=np.float32)
    sel = mask.astype(bool)
    out[sel] = (1 - alpha) * out[sel] + alpha * green
    return out.clip(0, 255).astype(np.uint8)


class LeafLensPredictor:
    """Loads the active model + label map and serves cached predictions."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = Registry(self.settings)
        self._model = None
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._load_active()

    def _load_active(self) -> None:
        files = self.registry.resolve_active()
        if files is None:
            raise FileNotFoundError(
                "No trained model registered. Run `python -m app.train` first."
            )
        self.version: str = str(files["version"])
        self.classes: list[str] = json.loads(files["labels"].read_text())
        self._model_path = files["model"]
        self._model = None  # (re)loaded lazily

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        import tensorflow as tf  # lazy

        from .losses import dice_bce_loss, dice_coefficient, iou_score

        return tf.keras.models.load_model(
            self._model_path,
            custom_objects={
                "dice_bce_loss": dice_bce_loss,
                "iou": iou_score,
                "iou_score": iou_score,
                "dice": dice_coefficient,
                "dice_coefficient": dice_coefficient,
            },
            compile=False,
        )

    def reload(self) -> str:
        """Re-resolve the active version, drop the loaded model + cache."""
        self._cache.clear()
        self._load_active()
        return self.version

    # --- prediction -------------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        size = self.settings.image_size
        rgb = image.convert("RGB").resize((size[1], size[0]))
        arr = np.asarray(rgb, dtype=np.float32)               # [0, 255]
        batch = arr[None, ...]

        out = self.model.predict(batch, verbose=0)
        seg = np.asarray(out["segmentation"])[0, ..., 0]      # HxW prob
        cls = np.asarray(out["classification"])[0]            # (num_classes,)

        order = np.argsort(cls)[::-1]
        k = min(self.settings.top_k, len(self.classes))
        top_k = [
            {"label": self.classes[i], "confidence": float(cls[i])}
            for i in order[:k]
        ]
        best = order[0]

        mask_bin = (seg >= self.settings.mask_threshold).astype(np.uint8)
        leaf_coverage = float(mask_bin.mean())
        mask_png = _png_base64((mask_bin * 255).astype(np.uint8))
        overlay = overlay_mask(arr.astype(np.uint8), mask_bin)
        overlay_png = _png_base64(overlay)

        return {
            "predicted_class": self.classes[best],
            "confidence": float(cls[best]),
            "top_k": top_k,
            "leaf_coverage": leaf_coverage,
            "mask_png_base64": mask_png,
            "overlay_png_base64": overlay_png,
        }

    def predict_bytes(self, raw: bytes) -> dict:
        """Cached prediction from raw image bytes (key = version + content hash)."""
        if self.settings.cache_size <= 0:
            return self.predict(Image.open(io.BytesIO(raw)))

        key = f"{self.version}:{hashlib.sha256(raw).hexdigest()}"
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        result = self.predict(Image.open(io.BytesIO(raw)))
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self.settings.cache_size:
            self._cache.popitem(last=False)
        return result

    def cache_stats(self) -> dict:
        return {
            "size": len(self._cache),
            "capacity": self.settings.cache_size,
            "hits": self._hits,
            "misses": self._misses,
        }


def artifact_exists(settings: Settings | None = None) -> bool:
    return Registry(settings or get_settings()).has_any()
